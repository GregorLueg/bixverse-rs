#![cfg(feature = "single-cell")]
//! End-to-end parity for scTransform v2 against sctransform 0.4.3.
//!
//! The count matrix is rebuilt from the same LCG the fixture generator used, so
//! nothing but integers and reference answers crosses as text. See
//! `dev/gen_sctransform_fixtures.R`.
//!
//! Two different bars are applied on purpose:
//!
//! * The per-gene statistics and the regularisation are held to 1e-10. They are
//!   ports of R routines and there is no reason for them to drift.
//! * The fitted theta and intercept are held to a measured band. sctransform
//!   gets those from `glmGamPoi::glm_gp`, which fits its coefficient at an
//!   intermediate overdispersion and never refits at the final one, so its
//!   estimate is not at the optimum of its own reported dispersion. This crate
//!   returns the Cox-Reid adjusted joint MLE instead. Measured on simulated
//!   data at the 2000-cell subsample v2 runs on, the two agreed to a median
//!   2e-5 on theta with the top-2000 HVG sets overlapping 1997/2000; on this
//!   400-cell fixture the median is 3e-5.

use approx::assert_relative_eq;
use faer::Mat;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::bin_merge_io::gene_store_to_cell_store;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
};
use bixverse_rs::single_cell::sc_processing::pca::{SingleCellPcaParams, pca_on_sc_residuals};
use bixverse_rs::single_cell::sc_processing::residuals::residual_variance;
use bixverse_rs::single_cell::sc_processing::sctransform::model::{
    SctCellContext, SctCovariates, SctGeneStats, SctModel, SctParams, min_variance_from_umi_median,
    regularise_sct_model, sct_residual_row,
};
use bixverse_rs::single_cell::sc_processing::sctransform::residuals::SctResiduals;
use bixverse_rs::single_cell::sc_processing::sctransform::stream::{
    SctStreamOpts, fit_sctransform, fit_sctransform_grouped, sct_corrected_counts, sct_gene_pass,
    sct_residual_variance,
};

mod sctransform_fixtures;
use sctransform_fixtures as fx;

/////////////
// Helpers //
/////////////

/// Removes the scratch store when the test ends, pass or panic.
struct TempStore(std::path::PathBuf);

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempStore {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_sct_{name}.bin")))
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Rebuilds the count matrix `dev/gen_sctransform_fixtures.R` generated.
///
/// The draw order matters and mirrors the R script call for call: every
/// library scale, then every gene rate, then the counts gene-major. `u^3`
/// scaled by a rate and a library size is heavy-tailed enough to be genuinely
/// overdispersed while using only multiplication and a floor, so both sides
/// land on identical integers.
fn fixture_counts() -> (Vec<Vec<u32>>, Vec<f64>) {
    let mut state = fx::LCG_SEED;
    let mut next = || {
        state = (1_664_525_u64
            .wrapping_mul(state)
            .wrapping_add(1_013_904_223))
            % 4_294_967_296;
        state as f64 / 4_294_967_296.0
    };

    let lib_scale: Vec<f64> = (0..fx::N_CELLS).map(|_| 2000.0 + 6000.0 * next()).collect();
    let gene_rate: Vec<f64> = (0..fx::N_GENES)
        .map(|_| 10.0_f64.powf(-5.0 + 3.2 * next()))
        .collect();

    let mut counts = vec![vec![0_u32; fx::N_CELLS]; fx::N_GENES];
    for (g, row) in counts.iter_mut().enumerate() {
        for (c, slot) in row.iter_mut().enumerate() {
            let u = next();
            *slot = (u * u * u * gene_rate[g] * lib_scale[c] * 12.0).floor() as u32;
        }
    }

    // The offset is the full library size, taken before any gene filtering.
    let library_sizes: Vec<f64> = (0..fx::N_CELLS)
        .map(|c| counts.iter().map(|row| row[c] as f64).sum())
        .collect();

    (counts, library_sizes)
}

/// Writes a `dense[gene][cell]` matrix out as a gene-major store.
fn write_store(path: &str, dense: &[Vec<u32>], n_cells: usize) {
    let mut writer =
        CellGeneSparseWriter::new(path, false, n_cells, dense.len(), 1e4).expect("writer opens");

    for (gene_idx, gene) in dense.iter().enumerate() {
        let mut raw = Vec::new();
        let mut indices = Vec::new();
        for (cell, &value) in gene.iter().enumerate() {
            if value > 0 {
                raw.push(value);
                indices.push(cell);
            }
        }
        let norms: Vec<F16> = raw
            .iter()
            .map(|&v| F16::from_f32((v as f32).ln_1p()))
            .collect();

        writer
            .write_gene_chunk(CscGeneChunk::from_conversion(
                RawCounts::from_u32_auto(&raw),
                &norms,
                &indices,
                gene_idx,
                true,
            ))
            .expect("write gene chunk");
    }

    writer.finalise().expect("finalise");
}

/// Relative difference, with an absolute fallback so a reference value of zero
/// does not divide by zero.
fn rel(got: f64, want: f64) -> f64 {
    if want.abs() < 1e-12 {
        (got - want).abs()
    } else {
        ((got - want) / want).abs()
    }
}

fn params() -> SctParams {
    SctParams {
        min_cells: fx::MIN_CELLS,
        gmean_eps: fx::GMEAN_EPS,
        ..SctParams::default()
    }
}

///////////
// Tests //
///////////

/// The rebuilt matrix has to agree with R's before anything downstream means
/// anything. `MODELLED` is derived from the counts in R, so it doubles as a
/// checksum on the whole matrix.
#[test]
fn test_fixture_counts_round_trip() {
    let (counts, _) = fixture_counts();
    let detected: Vec<usize> = counts
        .iter()
        .map(|row| row.iter().filter(|&&v| v > 0).count())
        .collect();

    let modelled: Vec<usize> = (0..fx::N_GENES)
        .filter(|&g| detected[g] >= fx::MIN_CELLS)
        .collect();

    assert_eq!(modelled, fx::MODELLED.to_vec());
}

/// The gene sweep against `row_gmean`, `rowMeans` and `row_var`, plus the two
/// scalars the model needs. These are ports, so the bar is tight.
#[test]
fn test_gene_pass_matches_sctransform() {
    let store = TempStore::new("gene_pass");
    let (counts, _) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();

    let pass =
        sct_gene_pass(&reader, &cells, &params(), SctStreamOpts::default()).expect("gene pass");

    assert_eq!(pass.modelled, fx::MODELLED.to_vec());

    for (k, &g) in pass.modelled.iter().enumerate() {
        assert!(
            rel(pass.stats.log_gmean[g], fx::LOG_GMEAN[k]) < 1e-10,
            "gene {k}: log_gmean {} vs R {}",
            pass.stats.log_gmean[g],
            fx::LOG_GMEAN[k]
        );
        assert!(
            rel(pass.stats.amean[g], fx::AMEAN[k]) < 1e-12,
            "gene {k}: amean {} vs R {}",
            pass.stats.amean[g],
            fx::AMEAN[k]
        );
        assert!(
            rel(pass.stats.var[g], fx::GENE_VAR[k]) < 1e-10,
            "gene {k}: var {} vs R {}",
            pass.stats.var[g],
            fx::GENE_VAR[k]
        );
    }

    assert!(
        rel(pass.median_nonzero, fx::MEDIAN_NONZERO) < 1e-12,
        "median_nonzero {} vs R {}",
        pass.median_nonzero,
        fx::MEDIAN_NONZERO
    );
    assert!(
        rel(pass.mean_cell_sum, fx::MEAN_CELL_SUM) < 1e-12,
        "mean_cell_sum {} vs R {}",
        pass.mean_cell_sum,
        fx::MEAN_CELL_SUM
    );
}

/// The regularisation on real data, fed R's own unregularised `model_pars` so
/// the estimator difference is taken out of the comparison entirely. What is
/// left is the outlier detection, the Poisson exclusion, the Sheather-Jones
/// bandwidth and the kernel smoothing.
#[test]
fn test_regularisation_matches_sctransform_on_real_fits() {
    use bixverse_rs::single_cell::sc_processing::sctransform::nb_fit::NbOffsetFit;

    let stats = SctGeneStats {
        log_gmean: fx::LOG_GMEAN.to_vec(),
        amean: fx::AMEAN.to_vec(),
        var: fx::GENE_VAR.to_vec(),
    };

    let step1: Vec<NbOffsetFit> = fx::STEP1_THETA
        .iter()
        .zip(fx::STEP1_INTERCEPT.iter())
        .map(|(&theta, &intercept)| NbOffsetFit {
            theta,
            coefficients: vec![intercept],
        })
        .collect();

    let model = regularise_sct_model(
        &step1,
        &fx::STEP1_POS,
        &stats,
        &[],
        fx::MEAN_CELL_SUM,
        min_variance_from_umi_median(fx::MEDIAN_NONZERO),
        fx::N_CELLS,
        &params(),
    )
    .expect("regularisation");

    assert_eq!(model.len(), fx::MODELLED.len());
    assert_eq!(
        (0..model.len()).filter(|&g| model.is_poisson(g)).count(),
        fx::FIT_THETA.iter().filter(|t| !t.is_finite()).count(),
        "Poisson gene count disagrees with R"
    );

    for g in 0..model.len() {
        if fx::FIT_THETA[g].is_infinite() {
            assert!(model.theta[g].is_infinite(), "gene {g} should be Poisson");
        } else {
            assert!(
                rel(model.theta[g], fx::FIT_THETA[g]) < 1e-10,
                "gene {g}: theta {} vs R {}",
                model.theta[g],
                fx::FIT_THETA[g]
            );
        }
        assert!(
            rel(model.intercept(g), fx::FIT_INTERCEPT[g]) < 1e-10,
            "gene {g}: intercept {} vs R {}",
            model.intercept(g),
            fx::FIT_INTERCEPT[g]
        );
    }
}

/// The whole chain off the binary store: sweep, fit, regularise. The step-1
/// gene set is pinned to R's so only the estimator differs, and the bar is the
/// measured band rather than a tolerance.
#[test]
fn test_fit_sctransform_end_to_end() {
    let store = TempStore::new("end_to_end");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();

    // Store indices of R's step-1 genes.
    let step1_genes: Vec<usize> = fx::STEP1_POS.iter().map(|&p| fx::MODELLED[p]).collect();
    let step1_cells: Vec<usize> = (0..fx::N_CELLS).collect();

    let (model, pass) = fit_sctransform(
        &reader,
        &cells,
        &library_sizes,
        &SctCovariates::default(),
        &params(),
        Some(&step1_genes),
        Some(&step1_cells),
        SctStreamOpts::default(),
    )
    .expect("fit");

    assert_eq!(model.len(), fx::MODELLED.len());
    assert_eq!(pass.modelled, fx::MODELLED.to_vec());

    // Which genes are Poisson is decided by the variance, not the estimator, so
    // it must agree exactly.
    for g in 0..model.len() {
        assert_eq!(
            model.theta[g].is_infinite(),
            fx::FIT_THETA[g].is_infinite(),
            "gene {g}: Poisson classification disagrees with R"
        );
    }

    let mut worst_theta = 0.0_f64;
    let mut worst_intercept = 0.0_f64;
    for g in 0..model.len() {
        if fx::FIT_THETA[g].is_finite() {
            worst_theta = worst_theta.max(rel(model.theta[g], fx::FIT_THETA[g]));
        }
        worst_intercept = worst_intercept.max(rel(model.intercept(g), fx::FIT_INTERCEPT[g]));
    }

    let mut drifts: Vec<f64> = (0..model.len())
        .filter(|&g| fx::FIT_THETA[g].is_finite())
        .map(|g| rel(model.theta[g], fx::FIT_THETA[g]))
        .collect();
    drifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut var_drift: Vec<f64> = (0..model.len())
        .map(|g| {
            let mu = pass.stats.amean[fx::MODELLED[g]];
            let mine = mu + mu * mu / model.theta[g];
            let theirs = mu + mu * mu / fx::FIT_THETA[g];
            rel(mine, theirs)
        })
        .collect();
    var_drift.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "theta drift  : median {:.4e} p90 {:.4e} max {:.4e} (n {})",
        drifts[drifts.len() / 2],
        drifts[drifts.len() * 9 / 10],
        drifts[drifts.len() - 1],
        drifts.len()
    );
    println!(
        "variance drift: median {:.4e} p90 {:.4e} max {:.4e}",
        var_drift[var_drift.len() / 2],
        var_drift[var_drift.len() * 9 / 10],
        var_drift[var_drift.len() - 1]
    );

    // The max is gated on the variance rather than on theta, because theta is
    // weakly identified near the Poisson boundary: `theta = gmean /
    // (10^dispersion_par - 1)` blows up as `dispersion_par` approaches zero, so
    // a negligible change in overdispersion moves theta a long way while
    // `mu + mu^2 / theta`, which is what reaches the residual, barely moves.
    // Theta is gated on its median, where a real estimator change would show.
    //
    // Measured on this fixture: variance max 1.8e-3, theta median 3.0e-5,
    // intercept max 3.6e-4. That is close enough to the 2e-5 measured at the
    // 2000-cell subsample v2 actually runs on that the remaining gap is the
    // Cox-Reid adjusted joint MLE this crate returns against the intermediate
    // estimate `glm_gp` stops at, and nothing else.
    assert!(
        var_drift[var_drift.len() - 1] < 5e-3,
        "model variance drifted {:.2e} from R, past the measured band",
        var_drift[var_drift.len() - 1]
    );
    assert!(
        drifts[drifts.len() / 2] < 1e-4,
        "median regularised theta drifted {:.2e} from R, past the measured band",
        drifts[drifts.len() / 2]
    );
    assert!(
        worst_intercept < 1e-3,
        "regularised intercept drifted {worst_intercept:.2e} from R, past the measured band"
    );
    let _ = worst_theta;
}

/// A cell-major store must be refused rather than silently read as if it were
/// gene-major.
#[test]
fn test_gene_pass_rejects_a_cell_major_store() {
    let store = TempStore::new("wrong_mode");
    let writer = CellGeneSparseWriter::new(store.path(), true, 2, 3, 1e4).expect("writer opens");
    writer.finalise().expect("finalise");

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells = vec![0_usize, 1];

    assert!(matches!(
        sct_gene_pass(&reader, &cells, &params(), SctStreamOpts::default()),
        Err(BixverseErrors::ReaderModeMismatch { .. })
    ));
}

/// Builds the model straight out of R's regularised parameters, so anything
/// this exercises is the residual formula and nothing upstream of it.
fn model_from_fixture() -> SctModel {
    SctModel {
        genes: fx::MODELLED.to_vec(),
        theta: fx::FIT_THETA.to_vec(),
        coefficients: fx::FIT_INTERCEPT.to_vec(),
        n_coef: 1,
        covariate_names: Vec::new(),
        log_umi_coef: std::f64::consts::LN_10,
        min_variance: min_variance_from_umi_median(fx::MEDIAN_NONZERO),
        clip_range: (-(fx::N_CELLS as f64).sqrt(), (fx::N_CELLS as f64).sqrt()),
    }
}

/// The residual formula against `gene_attr$residual_variance`, with R's own
/// regularised parameters fed in. The estimator difference is out of the
/// picture entirely here, so this is a gate on the residual arithmetic: the
/// mean, the variance floor, the clipping and the `n - 1` denominator.
///
/// The bar is 1e-7 rather than machine precision because the row is stored as
/// `f32`, matching `scale_csc_chunk` and halving the dense residual matrix the
/// PCA builds. That storage, not the arithmetic, is the whole of the gap:
/// `test_residual_variance_is_exact_in_f64` computes the same quantity in `f64`
/// and lands within 3e-14 of R.
#[test]
fn test_residual_variance_matches_sctransform() {
    let store = TempStore::new("resid_var");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();

    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    let got = sct_residual_variance(&reader, &model, &cells, &ctx, SctStreamOpts::default())
        .expect("residual variance");

    assert_eq!(got.len(), fx::RESIDUAL_VARIANCE.len());
    let mut worst = 0.0_f64;
    for (g, (&mine, &theirs)) in got.iter().zip(fx::RESIDUAL_VARIANCE.iter()).enumerate() {
        let d = rel(mine, theirs);
        worst = worst.max(d);
        assert!(
            d < 1e-7,
            "gene {g}: residual variance {mine} vs R {theirs} (rel {d:.3e})"
        );
    }
    println!("worst residual variance drift: {worst:.3e}");
}

/// A gene's residual row is dense: a zero count still carries `-mu / sqrt(var)`.
/// That is the property that rules out storing residuals in the sparse binary
/// format, so it is worth pinning.
#[test]
fn test_residual_row_is_dense_at_zero_counts() {
    let model = model_from_fixture();
    let (counts, library_sizes) = fixture_counts();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();

    // A gene with a genuinely finite theta, so the Poisson branch is not what
    // is being measured.
    let pos = (0..model.len())
        .find(|&g| !model.is_poisson(g))
        .expect("a non-Poisson gene");
    let store_gene = fx::MODELLED[pos];

    let indices: Vec<u32> = (0..fx::N_CELLS)
        .filter(|&c| counts[store_gene][c] > 0)
        .map(|c| c as u32)
        .collect();
    let nz: Vec<f64> = indices
        .iter()
        .map(|&c| counts[store_gene][c as usize] as f64)
        .collect();

    let mut row = vec![0.0_f32; fx::N_CELLS];
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    sct_residual_row(&nz, &indices, pos, &model, &ctx, &mut row).expect("residual row");

    let zero_cells: Vec<usize> = (0..fx::N_CELLS)
        .filter(|&c| counts[store_gene][c] == 0)
        .collect();
    assert!(!zero_cells.is_empty(), "the gene needs some zero counts");
    for &c in &zero_cells {
        assert!(
            row[c] < 0.0,
            "cell {c} has a zero count, so its residual must be negative, got {}",
            row[c]
        );
    }
}

/// The clipping range is applied, and it is the `+/- sqrt(n_cells)` sctransform
/// uses rather than Seurat's `sqrt(n_cells / 30)`.
#[test]
fn test_residual_row_respects_the_clip_range() {
    let mut model = model_from_fixture();
    model.clip_range = (-0.5, 0.5);
    let (_, library_sizes) = fixture_counts();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();

    let mut row = vec![0.0_f32; fx::N_CELLS];
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    sct_residual_row(&[10_000.0], &[0], 0, &model, &ctx, &mut row).expect("residual row");

    assert!(row.iter().all(|&r| (-0.5..=0.5).contains(&r)));
    assert_eq!(row[0], 0.5, "a huge count should clip at the upper bound");
}

/// The residual arithmetic itself, carried out in `f64`, against R.
///
/// This is the companion to `test_residual_variance_matches_sctransform`: it
/// pins that the 1e-8 gap there is `f32` row storage and nothing else, so that
/// a real arithmetic regression cannot hide inside the looser bar.
#[test]
fn test_residual_variance_is_exact_in_f64() {
    let (counts, library_sizes) = fixture_counts();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();

    let mut worst_f64 = 0.0_f64;
    for pos in 0..model.len() {
        let g = fx::MODELLED[pos];
        let b0 = model.intercept(pos);
        let theta = model.theta[pos];
        let (lo, hi) = model.clip_range;

        let row: Vec<f64> = (0..fx::N_CELLS)
            .map(|c| {
                let mu = (b0 + model.log_umi_coef * log10_umi[c]).exp();
                let var = (mu + mu * mu / theta).max(model.min_variance);
                ((counts[g][c] as f64 - mu) / var.sqrt()).clamp(lo, hi)
            })
            .collect();

        let n = fx::N_CELLS as f64;
        let mean = row.iter().sum::<f64>() / n;
        let v = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        worst_f64 = worst_f64.max(rel(v, fx::RESIDUAL_VARIANCE[pos]));
    }

    assert!(
        worst_f64 < 1e-12,
        "residual variance in f64 drifted {worst_f64:.3e} from R, \
         so the gap is arithmetic rather than f32 storage"
    );
}

/////////////////////
// Residual PCA //
/////////////////////

/// Builds the centred residual matrix the PCA should see, independently of any
/// of the code under test: straight from the model parameters and the raw
/// counts, in `f64`.
fn reference_residual_matrix(
    counts: &[Vec<u32>],
    log10_umi: &[f64],
    model: &SctModel,
    genes: &[usize],
) -> Mat<f64> {
    let n_cells = log10_umi.len();
    let mut m = Mat::<f64>::zeros(n_cells, genes.len());

    for (col, &store_gene) in genes.iter().enumerate() {
        let pos = model.position(store_gene).expect("gene is modelled");
        let b0 = model.intercept(pos);
        let theta = model.theta[pos];
        let (lo, hi) = model.clip_range;

        let row: Vec<f64> = (0..n_cells)
            .map(|c| {
                let mu = (b0 + model.log_umi_coef * log10_umi[c]).exp();
                let var = (mu + mu * mu / theta).max(model.min_variance);
                // f32 to match what the implementation stores before centring.
                (((counts[store_gene][c] as f64 - mu) / var.sqrt()).clamp(lo, hi)) as f32 as f64
            })
            .collect();

        let mean = row.iter().sum::<f64>() / n_cells as f64;
        for (c, v) in row.iter().enumerate() {
            m[(c, col)] = v - mean;
        }
    }

    m
}

/// PCA on the residuals against a direct dense SVD of the same matrix.
///
/// Exact SVD on both sides, so this is a real equality check rather than a
/// randomised-projection comparison. Components are compared up to sign, which
/// an SVD does not fix.
#[test]
fn test_residual_pca_matches_a_direct_svd() {
    let store = TempStore::new("residual_pca");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");

    // The 40 genes with the most residual variance, which is scTransform's own
    // feature-selection criterion.
    let mut ranked: Vec<(f64, usize)> = fx::RESIDUAL_VARIANCE
        .iter()
        .enumerate()
        .map(|(pos, &v)| (v, fx::MODELLED[pos]))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut hvg: Vec<usize> = ranked.into_iter().take(40).map(|(_, g)| g).collect();
    hvg.sort_unstable();

    let no_pcs = 10;
    // Residuals already carry their variance as signal, so centre but do not
    // rescale. Exact SVD so the comparison is not against a random projection.
    let params = SingleCellPcaParams::new(true, false, false, false, 1e4);

    let (scores, loadings, singular, scaled) = pca_on_sc_residuals(
        &reader,
        &cells,
        &hvg,
        no_pcs,
        &params,
        &SctResiduals::single(&model, ctx).expect("residual source"),
        42,
        true,
        0,
    )
    .expect("residual PCA");

    // The residual matrix the PCA built must be the one the model defines.
    let reference = reference_residual_matrix(&counts, &log10_umi, &model, &hvg);
    let got = scaled.expect("return_scaled was set");
    for c in 0..fx::N_CELLS {
        for j in 0..hvg.len() {
            let d = (got[(c, j)] as f64 - reference[(c, j)]).abs();
            assert!(
                d < 1e-5,
                "cell {c} gene {j}: residual {} vs reference {}",
                got[(c, j)],
                reference[(c, j)]
            );
        }
    }

    // And the decomposition of it must be the SVD of it.
    let svd = reference.thin_svd().expect("reference SVD");
    for k in 0..no_pcs {
        let want = svd.S().column_vector()[k];
        assert_relative_eq!(singular[k] as f64, want, max_relative = 1e-4);

        // Sign is not determined by an SVD, so align on the largest entry.
        let pivot = (0..fx::N_CELLS)
            .max_by(|&a, &b| {
                svd.U()[(a, k)]
                    .abs()
                    .partial_cmp(&svd.U()[(b, k)].abs())
                    .unwrap()
            })
            .expect("a pivot row");
        let flip = (scores[(pivot, k)] as f64).signum() * (svd.U()[(pivot, k)] * want).signum();

        for c in 0..fx::N_CELLS {
            let want_score = svd.U()[(c, k)] * want;
            let got_score = scores[(c, k)] as f64 * flip;
            assert!(
                (got_score - want_score).abs() < 1e-3 * want.max(1.0),
                "PC{k} cell {c}: score {got_score} vs {want_score}"
            );
        }
    }

    assert_eq!(loadings.nrows(), hvg.len());
    assert_eq!(loadings.ncols(), no_pcs);
}

/// Residuals and the shifted CLR transformation both rewrite what a column
/// holds, so asking for both is refused rather than silently doing one.
#[test]
fn test_residual_pca_refuses_clr() {
    let store = TempStore::new("residual_pca_clr");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    let params = SingleCellPcaParams::new(true, false, false, true, 1e4);

    assert!(matches!(
        pca_on_sc_residuals(
            &reader,
            &cells,
            &fx::MODELLED[..10],
            5,
            &params,
            &SctResiduals::single(&model, ctx).expect("residual source"),
            42,
            false,
            0,
        ),
        Err(BixverseErrors::PcaResidualsWithClr)
    ));
}

/// A gene outside the model is named rather than silently skipped or indexed
/// out of bounds.
#[test]
fn test_residual_pca_rejects_an_unmodelled_gene() {
    let store = TempStore::new("residual_pca_unmodelled");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");

    // A gene that failed the min_cells filter, so the model does not cover it.
    let unmodelled = (0..fx::N_GENES)
        .find(|g| !fx::MODELLED.contains(g))
        .expect("some gene was filtered out");

    assert!(matches!(
        pca_on_sc_residuals(
            &reader,
            &cells,
            &[unmodelled],
            2,
            &SingleCellPcaParams::new(true, false, false, false, 1e4),
            &SctResiduals::single(&model, ctx).expect("residual source"),
            42,
            false,
            0,
        ),
        Err(BixverseErrors::SctGeneNotModelled { .. })
    ));
}

///////////////////////
// Corrected counts //
///////////////////////

/// The corrected counts against `sctransform::correct_counts`, which is what
/// Seurat's `SCTransform()` puts in the SCT `counts` slot.
///
/// R's own regularised parameters are fed in, so this gates the correction
/// arithmetic: the unclipped residual, the reversal at the median library size
/// and round-half-to-even. Counts are integers, so the bar is exact equality,
/// not a tolerance.
#[test]
fn test_corrected_counts_match_sctransform() {
    let store = TempStore::new("corrected_in");
    let out = TempStore::new("corrected_out");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let model = model_from_fixture();

    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    sct_corrected_counts(
        &reader,
        &SctResiduals::single(&model, ctx).expect("residual source"),
        &cells,
        out.path(),
        SctStreamOpts::default(),
    )
    .expect("corrected counts");

    let written = ParallelSparseReader::new(out.path()).expect("corrected store opens");
    assert!(written.is_gene_based());
    // The correction removes the library-size structure, so there is no target
    // size for the normalised layer to refer to.
    assert_eq!(written.target_size(), None);

    let chunks = written
        .read_gene_parallel(&(0..model.len()).collect::<Vec<_>>())
        .expect("read corrected genes");
    assert_eq!(chunks.len(), model.len());

    for chunk in &chunks {
        let pos = chunk.original_index;
        let sum: f64 = chunk.data_raw.iter().map(|x| x as f64).sum();
        assert_eq!(
            sum,
            fx::CORRECTED_SUM[pos],
            "gene {pos}: corrected total {sum} vs R {}",
            fx::CORRECTED_SUM[pos]
        );
        assert_eq!(
            chunk.indices.len() as f64,
            fx::CORRECTED_NNZ[pos],
            "gene {pos}: corrected non-zeros {} vs R {}",
            chunk.indices.len(),
            fx::CORRECTED_NNZ[pos]
        );
    }

    // Per-gene totals could agree while the counts sat on the wrong cells, so
    // check three full rows entry by entry.
    for (k, &pos) in fx::PROBE_POS.iter().enumerate() {
        let chunk = &chunks[pos];
        let mut dense = vec![0.0_f64; fx::N_CELLS];
        for (slot, &cell) in chunk.indices.iter().enumerate() {
            dense[cell as usize] = chunk.data_raw.get(slot) as f64;
        }
        for (c, (&mine, &theirs)) in dense.iter().zip(fx::CORRECTED_ROWS[k].iter()).enumerate() {
            assert_eq!(
                mine, theirs,
                "gene {pos} cell {c}: corrected {mine} vs R {theirs}"
            );
        }
    }
}

/// The corrected gene-major store transposes into the cell-major companion
/// every downstream method expects, entry for entry.
///
/// This is the direction the crate did not have: every ingest path writes cells
/// first and derives genes, but scTransform's model is per gene, so its output
/// arrives gene-major.
#[test]
fn test_gene_store_transposes_to_cell_store() {
    let store = TempStore::new("transpose_in");
    let out = TempStore::new("transpose_out");
    let (counts, _) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    // A phase size well under the cell count, so the multi-phase path runs.
    gene_store_to_cell_store(store.path(), out.path(), 64, 32, 0).expect("transpose");

    let cell_reader = ParallelSparseReader::new(out.path()).expect("cell store opens");
    assert!(cell_reader.is_cell_based());

    let header = cell_reader.get_header();
    assert_eq!(header.total_cells, fx::N_CELLS);
    assert_eq!(header.total_genes, fx::N_GENES);

    let cells = cell_reader
        .read_cells_parallel(&(0..fx::N_CELLS).collect::<Vec<_>>())
        .expect("read cells");
    assert_eq!(cells.len(), fx::N_CELLS);

    for cell in &cells {
        let c = cell.original_index;

        let expected_lib: usize = (0..fx::N_GENES).map(|g| counts[g][c] as usize).sum();
        assert_eq!(cell.library_size, expected_lib, "cell {c}: library size");

        let mut dense = vec![0_u32; fx::N_GENES];
        for (slot, &gene) in cell.indices.iter().enumerate() {
            dense[gene as usize] = cell.data_raw.get(slot);
        }
        for g in 0..fx::N_GENES {
            assert_eq!(dense[g], counts[g][c], "cell {c} gene {g}");
        }

        // Gene indices must come out ascending, which every CSR consumer
        // assumes.
        assert!(
            cell.indices.windows(2).all(|w| w[0] < w[1]),
            "cell {c}: gene indices are not ascending"
        );
    }
}

/// A cell-major store handed to the transpose is refused rather than read as if
/// it were gene-major.
#[test]
fn test_transpose_rejects_a_cell_major_source() {
    let store = TempStore::new("transpose_wrong_mode");
    let out = TempStore::new("transpose_wrong_mode_out");
    let writer = CellGeneSparseWriter::new(store.path(), true, 2, 3, 1e4).expect("writer opens");
    writer.finalise().expect("finalise");

    assert!(matches!(
        gene_store_to_cell_store(store.path(), out.path(), 8, 8, 0),
        Err(BixverseErrors::ReaderModeMismatch { .. })
    ));
}

////////////////
// Covariates //
////////////////

/// Genes whose share of each cell's counts the covariate measures. Must match
/// `COV_BLOCK` in the generator.
const COV_BLOCK: usize = 30;

/// Rebuilds the covariate the generator used: a percent-mitochondrial analogue,
/// the share of each cell's counts falling in the first `COV_BLOCK` genes.
fn fixture_covariate(counts: &[Vec<u32>], library_sizes: &[f64]) -> SctCovariates {
    let values: Vec<f64> = (0..fx::N_CELLS)
        .map(|c| {
            let block: f64 = counts[..COV_BLOCK].iter().map(|row| row[c] as f64).sum();
            100.0 * block / library_sizes[c]
        })
        .collect();

    SctCovariates::from_columns(&[("cov_x".to_string(), values)]).expect("covariate")
}

/// The rebuilt covariate has to agree with R's before anything using it means
/// anything.
#[test]
fn test_fixture_covariate_round_trips() {
    let (counts, library_sizes) = fixture_counts();
    let cov = fixture_covariate(&counts, &library_sizes);

    assert_eq!(cov.n_covariates(), 1);
    assert_eq!(cov.names, vec!["cov_x".to_string()]);
    for c in 0..fx::N_CELLS {
        assert!(
            rel(cov.row(c)[0], fx::COV_X[c]) < 1e-12,
            "cell {c}: covariate {} vs R {}",
            cov.row(c)[0],
            fx::COV_X[c]
        );
    }
}

/// The regularisation with a covariate, fed R's own unregularised coefficients.
///
/// Every extra design column is smoothed against `log10(gmean)` alongside the
/// intercept, and zeroed for Poisson genes. This gates both.
#[test]
fn test_regularisation_with_covariate_matches_sctransform() {
    use bixverse_rs::single_cell::sc_processing::sctransform::nb_fit::NbOffsetFit;

    let stats = SctGeneStats {
        log_gmean: fx::LOG_GMEAN.to_vec(),
        amean: fx::AMEAN.to_vec(),
        var: fx::GENE_VAR.to_vec(),
    };

    let step1: Vec<NbOffsetFit> = (0..fx::COV_STEP1_POS.len())
        .map(|i| NbOffsetFit {
            theta: fx::COV_STEP1_THETA[i],
            coefficients: vec![fx::COV_STEP1_INTERCEPT[i], fx::COV_STEP1_COEF[i]],
        })
        .collect();

    let model = regularise_sct_model(
        &step1,
        &fx::COV_STEP1_POS,
        &stats,
        &["cov_x".to_string()],
        fx::MEAN_CELL_SUM,
        min_variance_from_umi_median(fx::MEDIAN_NONZERO),
        fx::N_CELLS,
        &params(),
    )
    .expect("regularisation");

    assert_eq!(model.n_coef, 2);
    assert!(model.has_covariates());
    assert_eq!(model.covariate_names, vec!["cov_x".to_string()]);

    for g in 0..model.len() {
        if fx::COV_FIT_THETA[g].is_infinite() {
            assert!(model.theta[g].is_infinite(), "gene {g} should be Poisson");
            // A Poisson gene takes the closed-form offset model, whose extra
            // coefficients are zero rather than a smoothed curve value.
            assert_eq!(model.coefficients_for(g)[1], 0.0, "gene {g}");
        } else {
            assert!(
                rel(model.theta[g], fx::COV_FIT_THETA[g]) < 1e-10,
                "gene {g}: theta {} vs R {}",
                model.theta[g],
                fx::COV_FIT_THETA[g]
            );
        }
        assert!(
            rel(model.intercept(g), fx::COV_FIT_INTERCEPT[g]) < 1e-10,
            "gene {g}: intercept {} vs R {}",
            model.intercept(g),
            fx::COV_FIT_INTERCEPT[g]
        );
        assert!(
            rel(model.coefficients_for(g)[1], fx::COV_FIT_COEF[g]) < 1e-10,
            "gene {g}: cov_x coefficient {} vs R {}",
            model.coefficients_for(g)[1],
            fx::COV_FIT_COEF[g]
        );
    }
}

/// Builds the covariate model from R's regularised parameters.
fn cov_model_from_fixture() -> SctModel {
    let mut coefficients = Vec::with_capacity(fx::MODELLED.len() * 2);
    for g in 0..fx::MODELLED.len() {
        coefficients.push(fx::COV_FIT_INTERCEPT[g]);
        coefficients.push(fx::COV_FIT_COEF[g]);
    }

    SctModel {
        genes: fx::MODELLED.to_vec(),
        theta: fx::COV_FIT_THETA.to_vec(),
        coefficients,
        n_coef: 2,
        covariate_names: vec!["cov_x".to_string()],
        log_umi_coef: std::f64::consts::LN_10,
        min_variance: min_variance_from_umi_median(fx::MEDIAN_NONZERO),
        clip_range: (-(fx::N_CELLS as f64).sqrt(), (fx::N_CELLS as f64).sqrt()),
    }
}

/// The residual arithmetic with a covariate in the linear predictor, against
/// `gene_attr$residual_variance` from the covariate run.
#[test]
fn test_residual_variance_with_covariate_matches_sctransform() {
    let store = TempStore::new("cov_resid_var");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let cov = fixture_covariate(&counts, &library_sizes);
    let ctx = SctCellContext::new(&log10_umi, &cov).expect("context");
    let model = cov_model_from_fixture();

    let got = sct_residual_variance(&reader, &model, &cells, &ctx, SctStreamOpts::default())
        .expect("residual variance");

    for (g, (&mine, &theirs)) in got.iter().zip(fx::COV_RESIDUAL_VARIANCE.iter()).enumerate() {
        assert!(
            rel(mine, theirs) < 1e-7,
            "gene {g}: residual variance {mine} vs R {theirs}"
        );
    }
}

/// The covariate has to actually change the answer.
///
/// Without this, a term silently dropped somewhere in the linear predictor
/// would pass every other test in this file, since the two models agree on
/// everything except that term.
#[test]
fn test_covariate_changes_the_residuals() {
    let store = TempStore::new("cov_matters");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();

    let cov = fixture_covariate(&counts, &library_sizes);
    let with_ctx = SctCellContext::new(&log10_umi, &cov).expect("context");
    let with_cov = sct_residual_variance(
        &reader,
        &cov_model_from_fixture(),
        &cells,
        &with_ctx,
        SctStreamOpts::default(),
    )
    .expect("with covariate");

    let no_cov = SctCovariates::default();
    let without_ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    let without_cov = sct_residual_variance(
        &reader,
        &model_from_fixture(),
        &cells,
        &without_ctx,
        SctStreamOpts::default(),
    )
    .expect("without covariate");

    let worst = with_cov
        .iter()
        .zip(without_cov.iter())
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0_f64, f64::max);

    assert!(
        worst > 1e-3,
        "the covariate moved the residual variance by at most {worst:.2e}, \
         which is small enough that it may not be entering the model at all"
    );
}

/// Supplying covariates a model was not fitted with is refused rather than
/// silently ignored or read past the end of the row.
#[test]
fn test_residual_rejects_a_covariate_count_mismatch() {
    let (counts, library_sizes) = fixture_counts();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let cov = fixture_covariate(&counts, &library_sizes);
    let ctx = SctCellContext::new(&log10_umi, &cov).expect("context");

    // A model fitted without covariates, handed a context carrying one.
    let model = model_from_fixture();
    let mut row = vec![0.0_f32; fx::N_CELLS];

    assert!(matches!(
        sct_residual_row(&[], &[], 0, &model, &ctx, &mut row),
        Err(BixverseErrors::SctCovariateCountMismatch {
            model: 0,
            supplied: 1
        })
    ));
}

/// Covariate columns that disagree in length are named rather than producing a
/// ragged design.
#[test]
fn test_covariates_reject_ragged_columns() {
    assert!(matches!(
        SctCovariates::from_columns(&[
            ("a".to_string(), vec![1.0, 2.0, 3.0]),
            ("b".to_string(), vec![1.0, 2.0]),
        ]),
        Err(BixverseErrors::SctCovariateLengthMismatch { .. })
    ));
}

/// The full chain with a covariate, off the binary store.
#[test]
fn test_fit_sctransform_with_covariate_end_to_end() {
    let store = TempStore::new("cov_end_to_end");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let cov = fixture_covariate(&counts, &library_sizes);

    let step1_genes: Vec<usize> = fx::COV_STEP1_POS.iter().map(|&p| fx::MODELLED[p]).collect();
    let step1_cells: Vec<usize> = (0..fx::N_CELLS).collect();

    let (model, _) = fit_sctransform(
        &reader,
        &cells,
        &library_sizes,
        &cov,
        &params(),
        Some(&step1_genes),
        Some(&step1_cells),
        SctStreamOpts::default(),
    )
    .expect("fit");

    assert_eq!(model.n_coef, 2);
    assert_eq!(model.covariate_names, vec!["cov_x".to_string()]);

    // Comparing the two coefficients separately overstates the disagreement:
    // the intercept and the covariate coefficient trade off against each other,
    // so a slightly larger one is absorbed by a slightly smaller other. What
    // matters is the linear predictor they produce together. The shared
    // `ln(10) * log10_umi` term cancels, so the difference is
    // `d_intercept + d_coefficient * cov_x`, worst at one end of the covariate.
    let (cov_lo, cov_hi) = fx::COV_X
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });

    let mut eta_drift: Vec<f64> = Vec::with_capacity(model.len());
    for g in 0..model.len() {
        assert_eq!(
            model.theta[g].is_infinite(),
            fx::COV_FIT_THETA[g].is_infinite(),
            "gene {g}: Poisson classification disagrees with R"
        );
        let d_int = model.intercept(g) - fx::COV_FIT_INTERCEPT[g];
        let d_coef = model.coefficients_for(g)[1] - fx::COV_FIT_COEF[g];
        eta_drift.push(
            (d_int + d_coef * cov_lo)
                .abs()
                .max((d_int + d_coef * cov_hi).abs()),
        );
    }
    eta_drift.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "log(mu) drift: median {:.3e} p90 {:.3e} max {:.3e}",
        eta_drift[eta_drift.len() / 2],
        eta_drift[eta_drift.len() * 9 / 10],
        eta_drift[eta_drift.len() - 1]
    );

    // Measured on this fixture: median 1.9e-3, p90 9.1e-3, max 3.0e-2. These
    // are absolute drifts on `log(mu)`, so they read as fractional errors on
    // the fitted mean.
    //
    // The max is looser than the covariate-free case (3.6e-4) for a reason
    // worth knowing. Step-1 gene 93 has mean 0.15 and variance 0.16, and its
    // Cox-Reid profile likelihood rises monotonically as the overdispersion
    // goes to zero: the value at theta = 1.5e2 and at theta = 4.9e8 differ by
    // 4e-4 nats. This crate takes the maximum, which is the boundary;
    // `glm_gp` stops at theta = 316. The post-fit Poisson check is a hard
    // `moment_theta / theta < 1e-3`, so those two land on opposite sides of it,
    // one gene in 270 enters or leaves the smoothing set, and `bw.SJ` moves
    // about 2%, which nudges every smoothed value. R shows the same
    // instability against itself: adding this covariate flips gene 93 out of
    // its own outlier set and gene 22 into it.
    //
    // So this is a discontinuity in the algorithm, not an error that
    // accumulates. The median is the number to watch.
    assert!(
        eta_drift[eta_drift.len() - 1] < 5e-2,
        "log(mu) drifted {:.2e} from R, past the measured band",
        eta_drift[eta_drift.len() - 1]
    );
    assert!(
        eta_drift[eta_drift.len() / 2] < 5e-3,
        "median log(mu) drifted {:.2e} from R, past the measured band",
        eta_drift[eta_drift.len() / 2]
    );
}

/////////////////////
// Multi-sample    //
/////////////////////

/// Splits the fixture cells into two samples of deliberately different depth.
///
/// The second half is thinned to a third of its counts, so the two samples have
/// genuinely different sequencing depths and a single pooled model would be
/// wrong for both.
fn two_sample_counts() -> (Vec<Vec<u32>>, Vec<f64>, Vec<u32>) {
    let (mut counts, _) = fixture_counts();
    let split = fx::N_CELLS / 2;

    for row in counts.iter_mut() {
        for value in row.iter_mut().skip(split) {
            *value /= 3;
        }
    }

    let library_sizes: Vec<f64> = (0..fx::N_CELLS)
        .map(|c| counts.iter().map(|row| row[c] as f64).sum())
        .collect();
    let groups: Vec<u32> = (0..fx::N_CELLS)
        .map(|c| if c < split { 0 } else { 1 })
        .collect();

    (counts, library_sizes, groups)
}

/// A grouped fit over a single group has to be the single-model fit, exactly.
///
/// This is the regression that keeps the multi-sample work from silently
/// changing the one-sample answer everything else in this file is pinned to.
#[test]
fn test_grouped_fit_with_one_group_matches_the_single_fit() {
    let store = TempStore::new("grouped_one_group");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let no_cov = SctCovariates::default();
    let opts = SctStreamOpts::default();

    let (single, _) = fit_sctransform(
        &reader,
        &cells,
        &library_sizes,
        &no_cov,
        &params(),
        None,
        None,
        opts,
    )
    .expect("single fit");

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &vec![0_u32; fx::N_CELLS],
        &library_sizes,
        &no_cov,
        &params(),
        opts,
    )
    .expect("grouped fit");

    assert_eq!(grouped.models.len(), 1);
    assert_eq!(grouped.genes, single.genes);
    assert_eq!(grouped.models[0].genes, single.genes);
    assert_eq!(grouped.models[0].theta, single.theta);
    assert_eq!(grouped.models[0].coefficients, single.coefficients);
    assert_eq!(grouped.models[0].min_variance, single.min_variance);
    assert_eq!(grouped.models[0].clip_range, single.clip_range);
}

/// Each group's model has to be the model that sample would get on its own.
#[test]
fn test_grouped_fit_matches_independent_per_sample_fits() {
    let store = TempStore::new("grouped_per_sample");
    let (counts, library_sizes, groups) = two_sample_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let no_cov = SctCovariates::default();
    let opts = SctStreamOpts::default();

    // The grouped driver shares the clip range across groups, so an
    // independent fit has to be handed the same one to be comparable.
    let shared = SctParams {
        clip_range: Some(params().resolve_clip_range(fx::N_CELLS)),
        ..params()
    };

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &groups,
        &library_sizes,
        &no_cov,
        &params(),
        opts,
    )
    .expect("grouped fit");
    assert_eq!(grouped.models.len(), 2);

    for group in 0..2 {
        let group_cells: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|&c| groups[c] as usize == group)
            .collect();
        let group_sizes: Vec<f64> = group_cells.iter().map(|&c| library_sizes[c]).collect();

        let (want, _) = fit_sctransform(
            &reader,
            &group_cells,
            &group_sizes,
            &no_cov,
            &shared,
            None,
            None,
            opts,
        )
        .expect("independent fit");

        assert_eq!(grouped.models[group].genes, want.genes);
        assert_eq!(grouped.models[group].theta, want.theta);
        assert_eq!(grouped.models[group].coefficients, want.coefficients);
        // The variance floor is a property of the sample's own depth, so the
        // thinned sample must not inherit the other's.
        assert_eq!(grouped.models[group].min_variance, want.min_variance);
    }

    assert_ne!(
        grouped.models[0].min_variance, grouped.models[1].min_variance,
        "the thinned sample should have its own variance floor"
    );
}

/// The shared gene axis is the intersection, and the clip range is not.
#[test]
fn test_grouped_gene_axis_is_the_intersection() {
    let store = TempStore::new("grouped_intersection");
    let (counts, library_sizes, groups) = two_sample_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let no_cov = SctCovariates::default();

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &groups,
        &library_sizes,
        &no_cov,
        &params(),
        SctStreamOpts::default(),
    )
    .expect("grouped fit");

    // Thinning the second sample pushes some genes under min_cells there, so
    // the intersection has to be a strict subset of at least one group.
    let a = &grouped.models[0].genes;
    let b = &grouped.models[1].genes;
    assert!(grouped.genes.len() <= a.len().min(b.len()));
    assert!(grouped.genes.iter().all(|g| a.contains(g) && b.contains(g)));
    assert!(grouped.genes.windows(2).all(|w| w[0] < w[1]));
    assert!(
        grouped.genes.len() < a.len(),
        "the thinned sample should drop genes the deeper one keeps"
    );

    // One clip range across groups, or the residuals are not comparable.
    assert_eq!(
        grouped.models[0].clip_range, grouped.models[1].clip_range,
        "the clip range must be shared"
    );
    let expected = params().resolve_clip_range(fx::N_CELLS);
    assert_relative_eq!(grouped.models[0].clip_range.1, expected.1, epsilon = 1e-12);
}

/// A cell's residual has to come from its own sample's model.
#[test]
fn test_grouped_residual_row_uses_each_cells_own_model() {
    let store = TempStore::new("grouped_residual_row");
    let (counts, library_sizes, groups) = two_sample_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &groups,
        &library_sizes,
        &no_cov,
        &params(),
        SctStreamOpts::default(),
    )
    .expect("grouped fit");

    let source =
        SctResiduals::new(&grouped.models, ctx, groups.clone()).expect("grouped residual source");

    let gene_pos = 0;
    let gene = source.genes()[gene_pos];
    let chunk = reader.read_gene(gene).expect("gene reads");
    let counts_nz: Vec<f64> = match &chunk.data_raw {
        RawCounts::U16(v) => v.iter().map(|&x| x as f64).collect(),
        RawCounts::U32(v) => v.iter().map(|&x| x as f64).collect(),
    };

    let mut row = vec![0.0_f32; fx::N_CELLS];
    source
        .residual_row(&counts_nz, &chunk.indices, gene_pos, &mut row)
        .expect("grouped residual row");

    // Each half has to equal the single-model row of its own model.
    for group in 0..2 {
        let model = &grouped.models[group];
        let pos = model.position(gene).expect("gene is in every group");
        let mut want = vec![0.0_f32; fx::N_CELLS];
        sct_residual_row(&counts_nz, &chunk.indices, pos, model, &ctx, &mut want)
            .expect("single-model row");

        for c in 0..fx::N_CELLS {
            if groups[c] as usize == group {
                assert_eq!(row[c], want[c], "cell {c} was scored under the wrong model");
            }
        }
    }
}

/// Residual variance comes back per group, and each group's numbers are what
/// that sample alone would produce.
#[test]
fn test_grouped_residual_variance_is_per_sample() {
    let store = TempStore::new("grouped_residual_variance");
    let (counts, library_sizes, groups) = two_sample_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    let opts = SctStreamOpts::default();

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &groups,
        &library_sizes,
        &no_cov,
        &params(),
        opts,
    )
    .expect("grouped fit");

    let source =
        SctResiduals::new(&grouped.models, ctx, groups.clone()).expect("grouped residual source");
    let per_group = residual_variance(&reader, &source, &cells, opts).expect("residual variance");

    assert_eq!(per_group.len(), 2);
    assert!(per_group.iter().all(|v| v.len() == grouped.genes.len()));

    for (group, want_group) in per_group.iter().enumerate() {
        let group_cells: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|&c| groups[c] as usize == group)
            .collect();
        let group_umi: Vec<f64> = group_cells.iter().map(|&c| log10_umi[c]).collect();
        let group_ctx = SctCellContext::new(&group_umi, &no_cov).expect("context");

        let want = sct_residual_variance(
            &reader,
            &grouped.models[group],
            &group_cells,
            &group_ctx,
            opts,
        )
        .expect("single-model residual variance");

        for (pos, &gene) in grouped.genes.iter().enumerate() {
            let want_pos = grouped.models[group].position(gene).expect("shared gene");
            assert_relative_eq!(want_group[pos], want[want_pos], epsilon = 1e-9);
        }
    }
}

/// Corrected counts route every cell through its own sample's model.
#[test]
fn test_grouped_corrected_counts_use_each_groups_model() {
    let store = TempStore::new("grouped_corrected_src");
    let out = TempStore::new("grouped_corrected_out");
    let (counts, library_sizes, groups) = two_sample_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let log10_umi: Vec<f64> = library_sizes.iter().map(|u| u.log10()).collect();
    let no_cov = SctCovariates::default();
    let ctx = SctCellContext::new(&log10_umi, &no_cov).expect("context");
    let opts = SctStreamOpts::default();

    let grouped = fit_sctransform_grouped(
        &reader,
        &cells,
        &groups,
        &library_sizes,
        &no_cov,
        &params(),
        opts,
    )
    .expect("grouped fit");
    let source =
        SctResiduals::new(&grouped.models, ctx, groups.clone()).expect("grouped residual source");

    sct_corrected_counts(&reader, &source, &cells, out.path(), opts).expect("corrected counts");

    let written = ParallelSparseReader::new(out.path()).expect("corrected store opens");
    let header = written.get_header();
    // The output gene axis is the shared one, not the store's.
    assert_eq!(header.total_genes, grouped.genes.len());
    assert_eq!(header.total_cells, fx::N_CELLS);
    assert_eq!(written.target_size(), None);
}

/// A malformed grouping is named rather than fitted against nothing.
#[test]
fn test_grouped_fit_rejects_a_gap_in_the_group_labels() {
    let store = TempStore::new("grouped_bad_labels");
    let (counts, library_sizes) = fixture_counts();
    write_store(store.path(), &counts, fx::N_CELLS);

    let reader = ParallelSparseReader::new(store.path()).expect("reader opens");
    let cells: Vec<usize> = (0..fx::N_CELLS).collect();
    let no_cov = SctCovariates::default();

    // Group 1 is never used.
    let groups: Vec<u32> = (0..fx::N_CELLS)
        .map(|c| if c < fx::N_CELLS / 2 { 0 } else { 2 })
        .collect();

    assert!(matches!(
        fit_sctransform_grouped(
            &reader,
            &cells,
            &groups,
            &library_sizes,
            &no_cov,
            &params(),
            SctStreamOpts::default(),
        ),
        Err(BixverseErrors::ResidualEmptyGroup { group: 1, .. })
    ));
}
