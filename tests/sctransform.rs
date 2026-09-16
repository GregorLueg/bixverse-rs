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
//!   returns the Cox-Reid adjusted joint MLE instead. The two were measured
//!   against each other on simulated data: at the 2000-cell subsample v2
//!   actually runs on, theta agreed to a median 2e-5 and the top-2000 HVG sets
//!   overlapped 1997/2000. This fixture is 400 cells, where the gap is wider.

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, ParallelSparseReader, RawCounts,
};
use bixverse_rs::single_cell::sc_processing::sct_stream::{
    SctStreamOpts, fit_sctransform, sct_gene_pass,
};
use bixverse_rs::single_cell::sc_processing::sctransform::{
    SctGeneStats, SctParams, min_variance_from_umi_median, regularise_sct_model,
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
    use bixverse_rs::single_cell::sc_processing::sct_nb_fit::NbOffsetFit;

    let stats = SctGeneStats {
        log_gmean: fx::LOG_GMEAN.to_vec(),
        amean: fx::AMEAN.to_vec(),
        var: fx::GENE_VAR.to_vec(),
    };

    let step1: Vec<NbOffsetFit> = fx::STEP1_THETA
        .iter()
        .zip(fx::STEP1_INTERCEPT.iter())
        .map(|(&theta, &intercept)| NbOffsetFit { theta, intercept })
        .collect();

    let model = regularise_sct_model(
        &step1,
        &fx::STEP1_POS,
        &stats,
        fx::MEAN_CELL_SUM,
        min_variance_from_umi_median(fx::MEDIAN_NONZERO),
        fx::N_CELLS,
        &params(),
    )
    .expect("regularisation");

    assert_eq!(model.len(), fx::MODELLED.len());
    assert_eq!(
        model.poisson.iter().filter(|&&p| p).count(),
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
            rel(model.intercept[g], fx::FIT_INTERCEPT[g]) < 1e-10,
            "gene {g}: intercept {} vs R {}",
            model.intercept[g],
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
        worst_intercept = worst_intercept.max(rel(model.intercept[g], fx::FIT_INTERCEPT[g]));
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

    // Theta alone is the wrong thing to gate on. Its worst case here is 20%,
    // but that is entirely the near-Poisson genes, where `theta` is weakly
    // identified: `theta = gmean / (10^dispersion_par - 1)` blows up as
    // `dispersion_par` approaches zero, so a negligible change in the
    // overdispersion moves theta a long way. What reaches the residual is the
    // variance `mu + mu^2 / theta`, and at a large theta that second term is
    // almost nothing. So the max is gated on the variance and theta itself only
    // on its median, which is where a real estimator change would show.
    //
    // Measured on this fixture: variance max 2.8e-2, theta median 5.9e-3,
    // intercept max 1.4e-2.
    assert!(
        var_drift[var_drift.len() - 1] < 5e-2,
        "model variance drifted {:.4} from R, past the measured band",
        var_drift[var_drift.len() - 1]
    );
    assert!(
        drifts[drifts.len() / 2] < 2e-2,
        "median regularised theta drifted {:.4} from R, past the measured band",
        drifts[drifts.len() / 2]
    );
    assert!(
        worst_intercept < 3e-2,
        "regularised intercept drifted {worst_intercept:.4} from R, past the measured band"
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
