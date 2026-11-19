//! Logistic regression differential testing, a port of Seurat's `LRDETest`
//! (`FindMarkers(test.use = "LR")`).
//!
//! Per feature, `group ~ x + latent` is tested against `group ~ latent` with a
//! binomial likelihood ratio test on one degree of freedom, `x` being the
//! normalised layer of the store and `latent` a per-cell covariate, usually
//! the library size. This is what the Signac vignettes recommend for
//! differential accessibility: the library size absorbs the depth difference
//! between the groups, which a Wilcoxon on the TF-IDF layer cannot. It does
//! nothing about cells being pseudoreplicates of their sample, so the
//! p-values still shrink with the number of cells.
//!
//! Both covariates are centred and scaled per fit. The likelihood ratio is
//! invariant to affine transformations of the design columns, and so are the
//! IRLS iterates, so this changes nothing but the conditioning: R fits the raw
//! library size.

use rayon::prelude::*;
use std::time::Instant;

use crate::prelude::*;
use crate::sc_analysis::dge_pathway_scores::{
    DgeAurocMultiRes, GENE_CHUNK_SIZE, GroupLayout, build_group_layout,
    calculate_dge_one_vs_many_auroc, simes_combine,
};
use crate::sc_analysis::fast_ranking::NO_GROUP;
use bixverse_core::core::math::distributions::chisq_sf;
use bixverse_core::core::math::stats::{TestAlternative, get_test_alternative, p_adjust_fdr};

///////////////
// Constants //
///////////////

/// Maximum IRLS iterations per fit, R's `glm.control(maxit = 25)`.
///
/// Matching R matters on separated features: the fit never converges there,
/// so the deviance, and with it the p-value, is whatever the last iterate
/// gives.
pub const LR_MAX_ITER: usize = 25;

/// Relative deviance change at which IRLS stops, R's
/// `glm.control(epsilon = 1e-8)` with its `|dev - dev_old| / (|dev| + 0.1)`
/// rule. Kept identical so a fit stops on the same iterate as `glm`.
pub const LR_DEVIANCE_TOL: f64 = 1e-8;

/// `|eta|` beyond which R's logit link clamps `mu` to `eps / (1 + eps)` and
/// `dmu / deta` to `eps` (`THRESH` in `family.c`). Replicated so separated
/// fits drift exactly as `glm`'s do.
const LOGIT_THRESH: f64 = 30.0;

/// Cholesky pivot, relative to its diagonal entry, below which the weighted
/// normal equations count as singular. Squared, this is the `1e-11`
/// tolerance `glm.fit` hands its pivoting QR. Singular means the feature is
/// collinear with the latent covariate, or every weight has collapsed onto
/// the clamp.
const PIVOT_TOL: f64 = 1e-22;

/// Coefficients of the largest model, intercept, feature and latent.
const MAX_COEF: usize = 3;

////////////////
// Structures //
////////////////

/// One feature's likelihood ratio test
#[derive(Clone, Copy, Debug)]
pub struct LrTest {
    /// `2 |ll_full - ll_null|`, i.e. the drop in deviance.
    pub stat: f64,
    /// Signed root of `stat`, positive when the feature is higher in group 1
    /// (the `y = 1` side). `z^2 = stat`, so the two-sided normal p-value of
    /// `z` is the chi-squared p-value of `stat`.
    pub z: f64,
    /// Whether the full model converged within [LR_MAX_ITER]. `false` almost
    /// always means (quasi-)separation, and the p-value is then the one of the
    /// last iterate, as in R. Separation can also pass as converged, once the
    /// deviance has shrunk to nothing, again as in R.
    pub converged: bool,
}

/// Structure to store the logistic regression DGE results in
///
/// Every vector except `genes_to_keep` has one entry per *kept* gene, in
/// increasing original gene index, as in
/// [crate::sc_analysis::dge_pathway_scores::DgeMannWhitneyRes].
#[derive(Clone, Debug)]
pub struct DgeLrRes {
    /// Difference of the mean normalised values, group 1 minus group 2, the
    /// same definition as the Mann-Whitney results.
    pub lfc: Vec<f32>,
    /// Proportions of cells in group 1 expressing the gene.
    pub prop1: Vec<f32>,
    /// Proportions of cells in group 2 expressing the gene.
    pub prop2: Vec<f32>,
    /// Likelihood ratio statistic, chi-squared on one degree of freedom.
    pub lr_stat: Vec<f64>,
    /// Signed root of the statistic, see [LrTest::z].
    pub z_scores: Vec<f64>,
    /// The p-values.
    pub p_vals: Vec<f64>,
    /// FDR values given the p-values.
    pub fdr: Vec<f64>,
    /// Whether the full model converged, see [LrTest::converged].
    pub converged: Vec<bool>,
    /// Boolean indicating if the gene was included in the analysis, i.e.,
    /// passed the proportion thresholds.
    pub genes_to_keep: Vec<bool>,
}

/// One comparison of the gene-wise scan: the groups on either side, its cells
/// and its null model.
struct LrArm {
    /// Groups on the `y = 1` side
    one: Vec<usize>,
    /// Groups on the `y = 0` side
    zero: Vec<usize>,
    /// Store cell indices, the `y = 1` cells first
    cells: Vec<usize>,
    /// `1.0` for the cells of `one`, `0.0` for those of `zero`
    y: Vec<f64>,
    /// The latent covariate over `cells`, centred and scaled
    latent: Vec<f64>,
    /// Deviance of `y ~ latent`
    null_dev: f64,
}

/// One gene's row of one arm: `(lfc, prop1, prop2, test)`.
type LrRow = (f32, f32, f32, LrTest);

/// One arm's scan output: the tested genes' rows and the gene mask.
type ArmScan = (Vec<LrRow>, Vec<bool>);

/// Per-thread scratch of the scan.
struct ScanScratch {
    /// The gene's normalised values over every store cell
    dense: Vec<f64>,
    /// The values over one arm's cells
    x: Vec<f64>,
    /// IRLS linear predictor
    eta: Vec<f64>,
    /// Non-zeros per group
    nnz: Vec<u64>,
    /// Sum of the normalised values per group
    sum: Vec<f64>,
}

/////////////
// Helpers //
/////////////

/// R's logit inverse link with its clamp, `binomial()$linkinv`
///
/// ### Params
///
/// * `eta` - Linear predictor
///
/// ### Returns
///
/// The fitted probability, clamped away from 0 and 1.
#[inline]
fn link_inv(eta: f64) -> f64 {
    let t = if eta < -LOGIT_THRESH {
        f64::EPSILON
    } else if eta > LOGIT_THRESH {
        1.0 / f64::EPSILON
    } else {
        eta.exp()
    };
    t / (1.0 + t)
}

/// R's `binomial()$mu.eta`, `dmu / deta` with the same clamp
///
/// ### Params
///
/// * `eta` - Linear predictor
///
/// ### Returns
///
/// The derivative of the mean with respect to `eta`.
#[inline]
fn mu_eta(eta: f64) -> f64 {
    if eta.abs() > LOGIT_THRESH {
        f64::EPSILON
    } else {
        let e = eta.exp();
        e / ((1.0 + e) * (1.0 + e))
    }
}

/// Binomial unit deviance of a 0/1 response, R's `binomial()$dev.resids`
///
/// ### Params
///
/// * `y` - Response, 0 or 1
/// * `mu` - Fitted probability
///
/// ### Returns
///
/// The unit deviance.
#[inline]
fn unit_deviance(y: f64, mu: f64) -> f64 {
    if y > 0.5 {
        2.0 * (1.0 / mu).ln()
    } else {
        2.0 * (1.0 / (1.0 - mu)).ln()
    }
}

/// Centre and scale in place
///
/// ### Params
///
/// * `v` - The values
///
/// ### Returns
///
/// `false`, leaving `v` centred only, when `v` is constant.
fn standardise(v: &mut [f64]) -> bool {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let ss: f64 = v.iter().map(|&a| (a - mean) * (a - mean)).sum();
    for a in v.iter_mut() {
        *a -= mean;
    }
    if ss <= 0.0 {
        return false;
    }
    let sd = (ss / n).sqrt();
    for a in v.iter_mut() {
        *a /= sd;
    }
    true
}

/// Solve the `p x p` symmetric positive definite system in place by Cholesky
///
/// ### Params
///
/// * `a` - The matrix, lower triangle read, overwritten by the factor
/// * `b` - Right-hand side, overwritten by the solution
/// * `p` - Dimension
///
/// ### Returns
///
/// `false` if a pivot falls below [PIVOT_TOL] relative to its diagonal entry.
fn cholesky_solve(a: &mut [[f64; MAX_COEF]; MAX_COEF], b: &mut [f64; MAX_COEF], p: usize) -> bool {
    for j in 0..p {
        let diag = a[j][j];
        let mut d = diag;
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        // also catches a NaN
        if d.partial_cmp(&(diag * PIVOT_TOL)) != Some(std::cmp::Ordering::Greater) {
            return false;
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in j + 1..p {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / d;
        }
    }
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= a[k][i] * b[k];
        }
        b[i] = s / a[i][i];
    }
    true
}

/// Logistic regression by IRLS, step for step `glm.fit` with the binomial
/// family and unit weights
///
/// Starts from `mu = (y + 0.5) / 2`, stops on [LR_DEVIANCE_TOL] or after
/// [LR_MAX_ITER] iterations. `glm.fit`'s step halving never triggers here:
/// the clamped link keeps every `mu` inside `(0, 1)` and the deviance finite.
/// A singular weighted design ends the fit at the previous iterate.
///
/// ### Params
///
/// * `y` - The 0/1 response
/// * `cols` - Design columns besides the intercept, at most two
/// * `eta` - Scratch for the linear predictor
///
/// ### Returns
///
/// `(deviance, coefficient of cols[0], converged)`. The coefficient is `0.0`
/// without columns.
fn irls_logistic(y: &[f64], cols: &[&[f64]], eta: &mut Vec<f64>) -> (f64, f64, bool) {
    let n = y.len();
    let p = 1 + cols.len();
    debug_assert!(p <= MAX_COEF);

    eta.clear();
    let mut dev_old = 0.0;
    for &yi in y {
        let mu = (yi + 0.5) / 2.0;
        let e = (mu / (1.0 - mu)).ln();
        eta.push(e);
        dev_old += unit_deviance(yi, link_inv(e));
    }

    let mut dev = dev_old;
    let mut coef = [0.0; MAX_COEF];

    for _ in 0..LR_MAX_ITER {
        let mut xtwx = [[0.0; MAX_COEF]; MAX_COEF];
        let mut xtwz = [0.0; MAX_COEF];
        let mut row = [1.0; MAX_COEF];

        for i in 0..n {
            let e = eta[i];
            let mu = link_inv(e);
            let me = mu_eta(e);
            let w = me * me / (mu * (1.0 - mu));
            let z = e + (y[i] - mu) / me;
            for (r, c) in row[1..p].iter_mut().zip(cols) {
                *r = c[i];
            }
            for j in 0..p {
                let wr = w * row[j];
                xtwz[j] += wr * z;
                for k in 0..=j {
                    xtwx[j][k] += wr * row[k];
                }
            }
        }

        if !cholesky_solve(&mut xtwx, &mut xtwz, p) {
            return (dev, coef[1], false);
        }
        coef = xtwz;

        dev = 0.0;
        for i in 0..n {
            let mut e = coef[0];
            for (j, c) in cols.iter().enumerate() {
                e += coef[j + 1] * c[i];
            }
            eta[i] = e;
            dev += unit_deviance(y[i], link_inv(e));
        }

        if (dev - dev_old).abs() / (dev.abs() + 0.1) < LR_DEVIANCE_TOL {
            return (dev, coef[1], true);
        }
        dev_old = dev;
    }

    (dev, coef[1], false)
}

/// The likelihood ratio test of one feature against a fitted null model
///
/// ### Params
///
/// * `y` - The 0/1 response
/// * `x` - The feature, centred and scaled in place
/// * `latent` - The latent covariate, already centred and scaled
/// * `null_dev` - Deviance of `y ~ latent`
/// * `eta` - IRLS scratch
///
/// ### Returns
///
/// The [LrTest]. A feature constant over the cells has nothing to test and
/// returns a zero statistic, i.e. `p = 1`. R aliases its coefficient instead,
/// and `lrtest` then reports zero degrees of freedom.
fn lr_against_null(
    y: &[f64],
    x: &mut [f64],
    latent: &[f64],
    null_dev: f64,
    eta: &mut Vec<f64>,
) -> LrTest {
    if !standardise(x) {
        return LrTest {
            stat: 0.0,
            z: 0.0,
            converged: true,
        };
    }
    let (dev, coef, converged) = irls_logistic(y, &[x, latent], eta);
    let stat = (null_dev - dev).abs();
    LrTest {
        stat,
        z: stat.sqrt().copysign(coef),
        converged,
    }
}

/// Seurat's `LRDETest` for one feature
///
/// Fits `y ~ x + latent` and `y ~ latent` by `glm`-identical IRLS and
/// returns the likelihood ratio between them.
///
/// ### Params
///
/// * `y` - Group membership, `true` for group 1
/// * `x` - The feature's normalised values
/// * `latent` - The latent covariate, e.g. the library size
///
/// ### Returns
///
/// The [LrTest]; its p-value is `chisq_sf(stat, 1)`, see [lr_p_value].
pub fn logistic_lr_test(y: &[bool], x: &[f64], latent: &[f64]) -> LrTest {
    let y: Vec<f64> = y.iter().map(|&b| f64::from(u8::from(b))).collect();
    let mut x = x.to_vec();
    let mut latent = latent.to_vec();
    let mut eta = Vec::with_capacity(y.len());
    let null_dev = null_deviance(&y, &mut latent, &mut eta);
    lr_against_null(&y, &mut x, &latent, null_dev, &mut eta)
}

/// P-value of a likelihood ratio test
///
/// Two-sided is the chi-squared tail of the statistic, which is what Seurat
/// reports. The one-sided alternatives halve it on the side the signed root
/// points to.
///
/// ### Params
///
/// * `test` - The test
/// * `alternative` - The alternative
///
/// ### Returns
///
/// The p-value.
pub fn lr_p_value(test: &LrTest, alternative: &TestAlternative) -> f64 {
    // only fails for df <= 0
    let p = chisq_sf(test.stat, 1.0).unwrap_or(f64::NAN);
    match alternative {
        TestAlternative::TwoSided => p,
        TestAlternative::Greater if test.z > 0.0 => p / 2.0,
        TestAlternative::Less if test.z < 0.0 => p / 2.0,
        _ => 1.0 - p / 2.0,
    }
}

/// Deviance of the null model `y ~ latent`, standardising `latent` in place
///
/// A constant latent covariate drops out and the null is the intercept only.
///
/// ### Params
///
/// * `y` - The 0/1 response
/// * `latent` - The latent covariate, standardised in place
/// * `eta` - Scratch for the linear predictor
///
/// ### Returns
///
/// The null deviance.
fn null_deviance(y: &[f64], latent: &mut [f64], eta: &mut Vec<f64>) -> f64 {
    if standardise(latent) {
        irls_logistic(y, &[latent], eta).0
    } else {
        irls_logistic(y, &[], eta).0
    }
}

/// Build one arm and fit its null model
///
/// ### Params
///
/// * `one` - Groups on the `y = 1` side
/// * `zero` - Groups on the `y = 0` side
/// * `groups` - The cell indices of each group
/// * `latent` - The latent covariate over every store cell
///
/// ### Returns
///
/// The [LrArm].
fn build_arm(one: Vec<usize>, zero: Vec<usize>, groups: &[Vec<usize>], latent: &[f64]) -> LrArm {
    let cells: Vec<usize> = one
        .iter()
        .chain(&zero)
        .flat_map(|&g| groups[g].iter().copied())
        .collect();
    let n1: usize = one.iter().map(|&g| groups[g].len()).sum();
    let y: Vec<f64> = (0..cells.len())
        .map(|i| if i < n1 { 1.0 } else { 0.0 })
        .collect();
    let mut lat: Vec<f64> = cells.iter().map(|&c| latent[c]).collect();
    let mut eta = Vec::with_capacity(cells.len());
    let null_dev = null_deviance(&y, &mut lat, &mut eta);
    LrArm {
        one,
        zero,
        cells,
        y,
        latent: lat,
        null_dev,
    }
}

/// Stream the gene store once and test every arm on every gene that passes
/// its filter
///
/// ### Params
///
/// * `reader` - Reader over the gene-based store
/// * `layout` - The cell groups
/// * `arms` - The comparisons
/// * `min_proportion` - Arm filter: the gene is tested if this share of
///   either side expresses it. Ignored with `gene_mask`
/// * `gene_mask` - A fixed gene filter shared by every arm
/// * `verbosity` - Prints decile progress at normal verbosity or above
///
/// ### Returns
///
/// Per arm, the rows of the tested genes and the gene mask.
fn scan_lr<S: SingleCellReading>(
    reader: &S,
    layout: &GroupLayout,
    arms: &[LrArm],
    min_proportion: f32,
    gene_mask: Option<&[bool]>,
    verbosity: Verbosity,
) -> Result<Vec<ArmScan>, ScErrors> {
    let dims = reader.dims();
    let n_groups = layout.sizes.len();
    let sizes = &layout.sizes;
    let all_genes: Vec<usize> = (0..dims.total_genes).collect();
    let mut out: Vec<ArmScan> = arms
        .iter()
        .map(|_| (Vec::new(), Vec::with_capacity(dims.total_genes)))
        .collect();
    let start = Instant::now();
    let mut done = 0;

    for block in all_genes.chunks(GENE_CHUNK_SIZE) {
        let chunks = reader.read_gene_parallel(block)?;

        let rows: Vec<Vec<Option<LrRow>>> = chunks
            .par_iter()
            .zip(block)
            .map_init(
                || ScanScratch {
                    dense: vec![0.0; dims.total_cells],
                    x: Vec::new(),
                    eta: Vec::new(),
                    nnz: vec![0; n_groups],
                    sum: vec![0.0; n_groups],
                },
                |s, (chunk, &gene)| {
                    let ScanScratch {
                        dense,
                        x,
                        eta,
                        nnz,
                        sum,
                    } = s;
                    nnz.fill(0);
                    sum.fill(0.0);
                    for (&cell, &v) in chunk.indices.iter().zip(&chunk.data_norm) {
                        let g = layout.lookup[cell as usize];
                        if g != NO_GROUP {
                            nnz[g as usize] += 1;
                            sum[g as usize] += v as f64;
                        }
                        dense[cell as usize] = v as f64;
                    }

                    let side = |gs: &[usize]| -> (f64, f64, usize) {
                        gs.iter().fold((0.0, 0.0, 0), |(nz, sm, n), &g| {
                            (nz + nnz[g] as f64, sm + sum[g], n + sizes[g])
                        })
                    };

                    let res = arms
                        .iter()
                        .map(|arm| {
                            let (nz1, sum1, n1) = side(&arm.one);
                            let (nz0, sum0, n0) = side(&arm.zero);
                            let (p1, p0) = ((nz1 / n1 as f64) as f32, (nz0 / n0 as f64) as f32);
                            let keep = match gene_mask {
                                Some(mask) => mask[gene],
                                None => p1 >= min_proportion || p0 >= min_proportion,
                            };
                            if !keep {
                                return None;
                            }
                            x.clear();
                            x.extend(arm.cells.iter().map(|&c| dense[c]));
                            let test = lr_against_null(&arm.y, x, &arm.latent, arm.null_dev, eta);
                            let lfc = (sum1 / n1 as f64 - sum0 / n0 as f64) as f32;
                            Some((lfc, p1, p0, test))
                        })
                        .collect();

                    for &cell in &chunk.indices {
                        dense[cell as usize] = 0.0;
                    }
                    res
                },
            )
            .collect();

        for gene_rows in rows {
            for ((arm_rows, mask), row) in out.iter_mut().zip(gene_rows) {
                mask.push(row.is_some());
                if let Some(row) = row {
                    arm_rows.push(row);
                }
            }
        }

        let prev = done;
        done += block.len();
        if verbosity.normal_verbosity() {
            report_decile_progress(done, prev, dims.total_genes, "genes", start.elapsed());
        }
    }

    Ok(out)
}

/// Assemble a [DgeLrRes] from an arm's rows
///
/// ### Params
///
/// * `rows` - The tested genes' rows
/// * `genes_to_keep` - The gene mask
/// * `alternative` - The test alternative
///
/// ### Returns
///
/// The [DgeLrRes].
fn build_lr_res(
    rows: Vec<LrRow>,
    genes_to_keep: Vec<bool>,
    alternative: &TestAlternative,
) -> DgeLrRes {
    let n = rows.len();
    let mut res = DgeLrRes {
        lfc: Vec::with_capacity(n),
        prop1: Vec::with_capacity(n),
        prop2: Vec::with_capacity(n),
        lr_stat: Vec::with_capacity(n),
        z_scores: Vec::with_capacity(n),
        p_vals: Vec::with_capacity(n),
        fdr: Vec::new(),
        converged: Vec::with_capacity(n),
        genes_to_keep,
    };
    for (lfc, p1, p2, test) in rows {
        res.lfc.push(lfc);
        res.prop1.push(p1);
        res.prop2.push(p2);
        res.lr_stat.push(test.stat);
        res.z_scores.push(test.z);
        res.p_vals.push(lr_p_value(&test, alternative));
        res.converged.push(test.converged);
    }
    res.fdr = p_adjust_fdr(&res.p_vals);
    res
}

/// Validate the latent covariate and the alternative
///
/// ### Params
///
/// * `latent` - The latent covariate
/// * `n_cells` - Cells in the store
/// * `alternative` - The alternative as given
///
/// ### Returns
///
/// The parsed alternative.
fn check_inputs(
    latent: &[f64],
    n_cells: usize,
    alternative: &str,
) -> Result<TestAlternative, ScErrors> {
    if latent.len() != n_cells {
        return Err(ScErrors::DgeLatentLength {
            len: latent.len(),
            n_cells,
        });
    }
    get_test_alternative(alternative)
        .ok_or_else(|| ScErrors::DgeUnknownAlternative(alternative.to_string()))
}

/// Fit the null models of the given arms and test them in one gene pass
///
/// ### Params
///
/// * `reader` - Reader over the gene-based store
/// * `groups` - The cell indices of each group
/// * `specs` - Per arm, the groups on the `y = 1` and the `y = 0` side
/// * `latent` - The latent covariate over every store cell
/// * `min_proportion` - The per-arm gene filter
/// * `alternative` - The test alternative
/// * `verbose` - Verbosity level
///
/// ### Returns
///
/// One [DgeLrRes] per arm.
fn run_arms<S: SingleCellReading>(
    reader: &S,
    groups: &[Vec<usize>],
    specs: Vec<(Vec<usize>, Vec<usize>)>,
    latent: &[f64],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<Vec<DgeLrRes>, ScErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let dims = reader.dims();
    let alternative = check_inputs(latent, dims.total_cells, alternative)?;
    let layout = build_group_layout(groups, dims.total_cells)?;
    let start = Instant::now();

    let arms: Vec<LrArm> = specs
        .into_par_iter()
        .map(|(one, zero)| build_arm(one, zero, groups, latent))
        .collect();
    let res = scan_lr(reader, &layout, &arms, min_proportion, None, verbosity)?
        .into_iter()
        .map(|(rows, mask)| build_lr_res(rows, mask, &alternative))
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "Logistic regression, {} arm(s) over {} groups: {:.2?}",
            arms.len(),
            groups.len(),
            start.elapsed()
        );
    }

    Ok(res)
}

//////////
// Main //
//////////

/// Get differential expression between two groups via logistic regression
///
/// Seurat's `LRDETest` with one latent covariate: per gene,
/// `group ~ x + latent` against `group ~ latent`, a likelihood ratio test on
/// one degree of freedom. The null model is fitted once, the full models in
/// parallel over the genes, reading the gene store once.
///
/// The gene filter is the Mann-Whitney one: a gene is tested if
/// `min_proportion` of either group expresses it. Seurat's `logfc.threshold`
/// and `min.diff.pct` filters are not applied, and the FDR is Benjamini
/// Hochberg rather than Seurat's Bonferroni.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `grp_1_indices` - The cell indices of group 1.
/// * `grp_2_indices` - The cell indices of group 2.
/// * `latent` - Per-cell covariate over every store cell, e.g. the library
///   size. Only the group cells are read.
/// * `min_proportion` - The minimum proportion that a gene needs to be
///   expressed in at least one of the two groups.
/// * `alternative` - `"twosided"` (Seurat's), or `"greater"` / `"less"` on
///   the sign of the feature coefficient.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The [DgeLrRes].
pub fn calculate_dge_grps_lr<S: SingleCellReading>(
    reader: &S,
    grp_1_indices: &[usize],
    grp_2_indices: &[usize],
    latent: &[f64],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<DgeLrRes, ScErrors> {
    let groups = [grp_1_indices.to_vec(), grp_2_indices.to_vec()];
    let specs = vec![(vec![0], vec![1])];
    run_arms(
        reader,
        &groups,
        specs,
        latent,
        min_proportion,
        alternative,
        verbose,
    )
    .map(|mut arms| arms.swap_remove(0))
}

/// Get one-vs-rest logistic regression differential expression for every
/// group
///
/// Each group against all other grouped cells pooled, see
/// [calculate_dge_grps_lr]. One null model per group, one pass over the gene
/// store for all of them. Each arm filters on its own, exactly as
/// [crate::sc_analysis::dge_pathway_scores::calculate_dge_one_vs_rest_mann_whitney]
/// does.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `groups` - The cell indices of each group. Cells in no group are ignored.
/// * `latent` - Per-cell covariate over every store cell.
/// * `min_proportion` - The minimum proportion that a gene needs to be
///   expressed in the group or in the rest.
/// * `alternative` - `"twosided"`, `"greater"` or `"less"`.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// One [DgeLrRes] per group, in input order. `prop1` is the group, `prop2`
/// the rest.
pub fn calculate_dge_one_vs_rest_lr<S: SingleCellReading>(
    reader: &S,
    groups: &[Vec<usize>],
    latent: &[f64],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<Vec<DgeLrRes>, ScErrors> {
    let specs = (0..groups.len())
        .map(|g| (vec![g], (0..groups.len()).filter(|&h| h != g).collect()))
        .collect();
    run_arms(
        reader,
        groups,
        specs,
        latent,
        min_proportion,
        alternative,
        verbose,
    )
}

/// One-vs-many markers with logistic regression p-values
///
/// [calculate_dge_one_vs_many_auroc] with every comparison's Mann-Whitney
/// test swapped for the likelihood ratio test of reference against rival
/// (reference as group 1), on the same global gene filter. The AUROC
/// summaries are unchanged; `z_scores` are the signed roots of the
/// statistics, and `p_vals`, `fdr`, `simes_p`, `max_p` and their FDRs are
/// recomputed from them. A second pass over the gene store.
///
/// ### Params
///
/// * `reader` - Reader over the gene-based count store.
/// * `groups` - The cell indices of each group.
/// * `references` - Indices into `groups` of the reference arms to report.
/// * `latent` - Per-cell covariate over every store cell.
/// * `min_proportion` - The minimum proportion that a gene needs to be
///   expressed in at least one of the groups.
/// * `alternative` - `"twosided"`, `"greater"` or `"less"`.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// One [DgeAurocMultiRes] per reference, in the order of `references`.
pub fn calculate_dge_one_vs_many_lr<S: SingleCellReading>(
    reader: &S,
    groups: &[Vec<usize>],
    references: &[usize],
    latent: &[f64],
    min_proportion: f32,
    alternative: &str,
    verbose: usize,
) -> Result<Vec<DgeAurocMultiRes>, ScErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let dims = reader.dims();
    let alt = check_inputs(latent, dims.total_cells, alternative)?;
    let mut res = calculate_dge_one_vs_many_auroc(
        reader,
        groups,
        references,
        min_proportion,
        alternative,
        verbose,
    )?;
    let layout = build_group_layout(groups, dims.total_cells)?;
    let n_groups = groups.len();
    let pairs: Vec<(usize, usize)> = references
        .iter()
        .flat_map(|&r| (0..n_groups).filter(move |&k| k != r).map(move |k| (r, k)))
        .collect();

    let mask = match res.first() {
        Some(first) => first.genes_to_keep.clone(),
        None => return Ok(res),
    };
    if !mask.iter().any(|&k| k) {
        return Ok(res);
    }

    let arms: Vec<LrArm> = pairs
        .par_iter()
        .map(|&(r, k)| build_arm(vec![r], vec![k], groups, latent))
        .collect();
    let mut scanned = scan_lr(
        reader,
        &layout,
        &arms,
        min_proportion,
        Some(&mask),
        verbosity,
    )?
    .into_iter();

    for arm in res.iter_mut() {
        for c in 0..n_groups - 1 {
            let (rows, _) = scanned
                .next()
                .expect("one scanned arm per reference and rival");
            let lr = build_lr_res(rows, Vec::new(), &alt);
            arm.z_scores[c] = lr.z_scores;
            arm.p_vals[c] = lr.p_vals;
            arm.fdr[c] = lr.fdr;
        }
        let n_genes = arm.prop_ref.len();
        let (simes_p, max_p): (Vec<f64>, Vec<f64>) = (0..n_genes)
            .map(|g| {
                let mut p: Vec<f64> = arm.p_vals.iter().map(|c| c[g]).collect();
                let max_p = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (simes_combine(&mut p), max_p)
            })
            .unzip();
        arm.simes_fdr = p_adjust_fdr(&simes_p);
        arm.max_p_fdr = p_adjust_fdr(&max_p);
        arm.simes_p = simes_p;
        arm.max_p = max_p;
    }

    Ok(res)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The signed root carries the direction and the one-sided p-values
    /// split the two-sided one.
    #[test]
    fn test_lr_direction_and_alternatives() {
        let n = 200;
        let y: Vec<bool> = (0..n).map(|i| i < 100).collect();
        // higher in group 1, noisy, latent unrelated
        let x: Vec<f64> = (0..n)
            .map(|i| (if i < 100 { 1.0 } else { 0.0 }) + ((i * 37 % 17) as f64) / 8.0)
            .collect();
        let latent: Vec<f64> = (0..n).map(|i| 1000.0 + (i * 53 % 29) as f64).collect();

        let test = logistic_lr_test(&y, &x, &latent);
        assert!(test.converged);
        assert!(test.z > 0.0);
        assert_relative_eq!(test.z * test.z, test.stat, max_relative = 1e-12);

        let two = lr_p_value(&test, &TestAlternative::TwoSided);
        let greater = lr_p_value(&test, &TestAlternative::Greater);
        let less = lr_p_value(&test, &TestAlternative::Less);
        assert_relative_eq!(greater, two / 2.0);
        assert_relative_eq!(less, 1.0 - two / 2.0);
    }

    /// A feature constant over the cells has nothing to test.
    #[test]
    fn test_lr_constant_feature() {
        let y: Vec<bool> = (0..20).map(|i| i < 10).collect();
        let x = vec![0.0; 20];
        let latent: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let test = logistic_lr_test(&y, &x, &latent);
        assert_eq!(test.stat, 0.0);
        assert_eq!(lr_p_value(&test, &TestAlternative::TwoSided), 1.0);
    }

    /// Perfect separation drives the full deviance to zero, deterministically,
    /// leaving the null deviance's tail as the p-value.
    #[test]
    fn test_lr_separation_is_deterministic() {
        let y: Vec<bool> = (0..40).map(|i| i < 20).collect();
        let x: Vec<f64> = (0..40).map(|i| if i < 20 { 1.0 } else { 0.0 }).collect();
        let latent: Vec<f64> = (0..40).map(|i| 50.0 + (i * 7 % 11) as f64).collect();
        let a = logistic_lr_test(&y, &x, &latent);
        let b = logistic_lr_test(&y, &x, &latent);
        assert_eq!(a.stat.to_bits(), b.stat.to_bits());
        assert!(lr_p_value(&a, &TestAlternative::TwoSided) < 1e-10);
    }

    /// A latent covariate of the wrong length is rejected.
    #[test]
    fn test_check_inputs() {
        assert!(matches!(
            check_inputs(&[1.0, 2.0], 3, "twosided"),
            Err(ScErrors::DgeLatentLength { len: 2, n_cells: 3 })
        ));
        assert!(matches!(
            check_inputs(&[1.0, 2.0, 3.0], 3, "both"),
            Err(ScErrors::DgeUnknownAlternative(_))
        ));
    }
}
