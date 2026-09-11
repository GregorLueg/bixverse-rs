#![cfg(feature = "single-cell")]

//! CellSweep parity against the reference implementation.
//!
//! The count matrix is rebuilt here from the same LCG the generator used, so
//! no float input crosses as text. Only the reference's outputs live in
//! `cellsweep_fixtures`.
//!
//! Parity is to a tolerance rather than bit-exact, and the tolerance is not
//! set by machine precision. The reference's per-row log-likelihood, `A_n` and
//! `alpha` accumulate in `f32` where this crate uses `f64`, which perturbs the
//! convergence trace. That only matters because at the reference's default
//! stopping rule the parameters are still drifting when it fires, so the two
//! implementations sample the same trajectory at slightly different points.
//! `beta` is the extreme case: it reads 1.3e-4 at the default tolerance and
//! 6.4e-12 once driven to the fixed point, so it is not converged at all.
//!
//! Hence two gates. The `tight_tol` fixture drives both sides to the fixed
//! point and is held to a tight tolerance; the `default_tol` fixture checks
//! the stopping rule itself and is necessarily looser.

mod cellsweep_fixtures;

use approx::assert_relative_eq;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_processing::cellsweep::{
    CellSweepFit, CellSweepParams, CellSweepSample, run_cellsweep,
};

use cellsweep_fixtures as fx;

/// Relative tolerance at the fixed point.
///
/// Both sides have stopped moving here, so what is left is the `f32` versus
/// `f64` accumulator difference propagated through the profiles.
const TIGHT_TOL: f64 = 1e-4;

/// Relative tolerance at the reference's default stopping rule.
///
/// Bounded by how far the parameters are still drifting when the rule fires,
/// not by precision. The near-zero `alpha` values are the worst case, since
/// `alpha = A_n / (A_n + C_n)` amplifies the relative error as `A_n` goes to
/// zero, so those carry an absolute floor instead. Every one of them agrees to
/// [TIGHT_TOL] at the fixed point.
const DEFAULT_TOL: f64 = 1e-2;

/// Relative tolerance on the recovered denoised column margins.
///
/// Not a model tolerance. The margins are reconstructed out of the f16
/// normalised layer scaled by the rounded library size, so the budget is f16
/// resolution plus unbiased rounding noise plus the sub-integer entries that
/// rounded away and left the layer. Measured max over the 60 fixture genes is
/// 7.8e-3, so this leaves about a quarter headroom. Loosening it much further
/// stops it constraining the per-entry split of a barcode's denoised mass,
/// which is the only thing this assertion is here for.
const COL_MARGIN_TOL: f64 = 1e-2;

/// Numerical Recipes LCG, mirroring `lcg()` in the fixture generator.
struct Lcg(u32);

impl Lcg {
    /// Advance the generator.
    ///
    /// ### Returns
    ///
    /// The new state, which is also the drawn value.
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
}

/// Ambient profile of the fixture, skewed towards the first marker block.
///
/// ### Returns
///
/// Per-gene ambient weights, summing to one.
fn ambient_shape() -> Vec<f64> {
    let block = fx::N_GENES / fx::N_CELLTYPES;
    let weights: Vec<f64> = (0..fx::N_GENES)
        .map(|g| if g < block { fx::AMBIENT_SKEW } else { 1.0 })
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

/// Rebuild the fixture's count matrix and cell-type labels.
///
/// Must stay in lockstep with `build_counts()` in the generator: the same LCG
/// in the same draw order, and the same `int(round(...))` rounding, which is
/// round-half-to-even in Python.
///
/// ### Returns
///
/// Dense rows (real barcodes first, then empty droplets) and the cell-type
/// code of each real barcode.
fn build_counts() -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut rng = Lcg(fx::LCG_SEED);
    let block = fx::N_GENES / fx::N_CELLTYPES;
    let ambient = ambient_shape();

    let mut rows = Vec::with_capacity(fx::N_REAL + fx::N_EMPTY);
    let mut labels = Vec::with_capacity(fx::N_REAL);

    for i in 0..fx::N_REAL {
        let k = i % fx::N_CELLTYPES;
        labels.push(k);
        let budget = 20.0 + (rng.next() % 300) as f64;
        let row = (0..fx::N_GENES)
            .map(|g| {
                let r = rng.next() % 8;
                let own = if k * block <= g && g < (k + 1) * block {
                    30 + r
                } else {
                    0
                };
                own + round_half_even(budget * ambient[g])
            })
            .collect();
        rows.push(row);
    }

    for _ in 0..fx::N_EMPTY {
        let total = 200.0 + (rng.next() % 200) as f64;
        let row = (0..fx::N_GENES)
            .map(|g| round_half_even(total * ambient[g]))
            .collect();
        rows.push(row);
    }

    (rows, labels)
}

/// Round half to even, matching Python's `round`.
///
/// `f64::round` rounds half away from zero, which would put a gene one count
/// out whenever the product lands exactly on `.5`.
///
/// ### Params
///
/// * `x` - Non-negative value to round.
///
/// ### Returns
///
/// The rounded value.
fn round_half_even(x: f64) -> u32 {
    let floor = x.floor();
    let frac = x - floor;
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    if round_up {
        floor as u32 + 1
    } else {
        floor as u32
    }
}

/// RAII guard that removes a test's temp file even if an assert fails.
struct TempBin(std::path::PathBuf);

impl Drop for TempBin {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl TempBin {
    /// Reserve a uniquely named scratch file in the system temp directory.
    ///
    /// ### Params
    ///
    /// * `name` - Test-unique suffix.
    ///
    /// ### Returns
    ///
    /// The guard; the path is available via [`Self::path`].
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bixverse_cellsweep_parity_{name}.bin")))
    }

    /// Path of the guarded file as a `&str`.
    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

/// Run CellSweep over the fixture and hand back the fit and the output store.
///
/// ### Params
///
/// * `name` - Test-unique suffix for the scratch files.
/// * `params` - Model parameters to run with.
///
/// ### Returns
///
/// The fit, plus a reader over the denoised store. The temp guards are
/// returned alongside so the files outlive the reader.
fn run_fixture(
    name: &str,
    params: CellSweepParams,
) -> (CellSweepFit, ParallelSparseReader, TempBin, TempBin) {
    let (rows, labels) = build_counts();
    let raw = TempBin::new(&format!("{name}_raw"));
    let out = TempBin::new(&format!("{name}_out"));

    let mut writer = CellGeneSparseWriter::new(raw.path(), true, rows.len(), fx::N_GENES, 1e4)
        .expect("writer opens");
    for (i, row) in rows.iter().enumerate() {
        let indices: Vec<u32> = (0..fx::N_GENES as u32)
            .filter(|&g| row[g as usize] > 0)
            .collect();
        let counts: Vec<u32> = indices.iter().map(|&g| row[g as usize]).collect();
        writer
            .write_cell_chunk(CsrCellChunk::from_data(&counts, &indices, i, 1e4, true))
            .expect("chunk writes");
    }
    writer.finalise().expect("store finalises");

    let reader = ParallelSparseReader::new(raw.path()).expect("store opens");
    let sample = CellSweepSample {
        sample_id: "fixture".to_string(),
        real_cells: (0..fx::N_REAL).collect(),
        empty_cells: (fx::N_REAL..fx::N_REAL + fx::N_EMPTY).collect(),
        celltype_idx: labels,
        n_celltypes: fx::N_CELLTYPES,
    };

    let mut run =
        run_cellsweep(&reader, &[sample], params, out.path(), 1e4, 0).expect("cellsweep runs");
    let denoised = ParallelSparseReader::new(out.path()).expect("output opens");

    (run.fits.remove(0), denoised, raw, out)
}

/// Recover the denoised row and column margins from the normalised layer.
///
/// The raw layer holds stochastically rounded counts, which cannot be matched
/// against numpy's generator, so the float values are read back out of the
/// normalised layer instead. `ln1p(v / total * target)` is inverted per entry
/// and the row is rescaled to the reference's row sum, which makes the column
/// margins the actual assertion: they only line up if the per-entry split of
/// each barcode's denoised mass matches.
///
/// ### Params
///
/// * `denoised` - Reader over the denoised store.
///
/// ### Returns
///
/// Row margins and column margins.
fn recover_col_margins(denoised: &ParallelSparseReader) -> Vec<f64> {
    let cells = denoised.get_all_cells().expect("cells read back");
    let mut col_sums = vec![0.0_f64; fx::N_GENES];

    for cell in &cells {
        let fractions: Vec<f64> = cell
            .data_norm
            .iter()
            .map(|v| v.to_f64().exp_m1() / 1e4)
            .collect();
        let total: f64 = fractions.iter().sum();
        if total <= 0.0 {
            continue;
        }

        // Scaled by the store's own rounded library size, not by a row total
        // taken from the fixture. Feeding the reference's row sums back in
        // would make the row margins trivially self-consistent and would also
        // inflate every column, since the normalised layer no longer holds the
        // sub-integer entries that rounded away.
        let library = cell.library_size as f64;
        for (&gene, fraction) in cell.indices.iter().zip(&fractions) {
            col_sums[gene as usize] += fraction / total * library;
        }
    }

    col_sums
}

/// Rounded library size per written barcode, from the raw layer.
///
/// The only absolute quantity the store carries. The normalised layer holds a
/// within-row distribution and nothing else, since `ln1p(v / total * target)`
/// throws the row total away.
///
/// ### Params
///
/// * `denoised` - Reader over the denoised store.
///
/// ### Returns
///
/// Library size per barcode, in output order.
fn raw_library_sizes(denoised: &ParallelSparseReader) -> Vec<f64> {
    denoised
        .get_all_cells()
        .expect("cells read back")
        .iter()
        .map(|cell| cell.library_size as f64)
        .collect()
}

#[test]
fn test_rebuilt_counts_match_the_reference_input() {
    // Guards the whole fixture: if the LCG or the rounding drifts, every
    // parity assertion below would be comparing against a different matrix.
    let (rows, _) = build_counts();

    let row_sums: Vec<usize> = rows[..fx::N_REAL]
        .iter()
        .map(|row| row.iter().map(|&c| c as usize).sum())
        .collect();
    assert_eq!(row_sums, fx::RAW_ROW_SUMS.to_vec());

    let col_sums: Vec<usize> = (0..fx::N_GENES)
        .map(|g| {
            rows[..fx::N_REAL]
                .iter()
                .map(|row| row[g] as usize)
                .sum::<usize>()
        })
        .collect();
    assert_eq!(col_sums, fx::RAW_COL_SUMS.to_vec());
}

#[test]
fn test_cellsweep_matches_the_reference_at_the_fixed_point() {
    use fx::tight_tol as want;

    let params = CellSweepParams {
        max_iter: want::MAX_ITER,
        del0_ll_tol: want::DEL0_LL_TOL,
        min_ll_tol: want::MIN_LL_TOL,
        tol_p: want::TOL_P as f32,
        tol_f: want::TOL_F,
        ..Default::default()
    };
    let (fit, denoised, _raw, _out) = run_fixture("tight", params);

    assert_relative_eq!(
        fit.log_likelihood,
        want::LOG_LIKELIHOOD,
        max_relative = TIGHT_TOL
    );

    // Both sides drive beta to zero here, so relative comparison is
    // meaningless; what matters is that it is negligible on both.
    assert!(
        fit.beta.abs() < 1e-8,
        "beta = {} did not collapse at the fixed point",
        fit.beta
    );

    assert_eq!(fit.alpha.len(), want::ALPHA.len());
    for (i, (&got, &target)) in fit.alpha.iter().zip(want::ALPHA.iter()).enumerate() {
        assert_relative_eq!(got, target, max_relative = TIGHT_TOL, epsilon = 1e-9);
        assert!(
            (0.0..=1.0).contains(&got),
            "alpha[{i}] = {got} is out of range"
        );
    }

    assert_eq!(fit.z_hat, want::Z_HAT.to_vec());

    assert_eq!(fit.ambient.len(), want::AMBIENT.len());
    for (&got, &target) in fit.ambient.iter().zip(want::AMBIENT.iter()) {
        assert_relative_eq!(got as f64, target, max_relative = TIGHT_TOL, epsilon = 1e-9);
    }

    assert_eq!(fit.celltype_profiles.len(), want::PROFILES.len());
    for (&got, &target) in fit.celltype_profiles.iter().zip(want::PROFILES.iter()) {
        assert_relative_eq!(got as f64, target, max_relative = TIGHT_TOL, epsilon = 1e-9);
    }

    // -- the denoised matrix itself --
    //
    // Row margins deliberately are not asserted against the reference here.
    // The normalised layer only carries a within-row distribution, so any
    // absolute recovery has to be handed a row total from outside, and
    // asserting that total back against the value used to produce it is
    // vacuous. What is checked instead: the rounded library sizes, which are
    // an independent absolute quantity, and the column margins, which only
    // line up if the within-row split matches.
    let libs = raw_library_sizes(&denoised);
    let got_total: f64 = libs.iter().sum();
    let want_total: f64 = want::DENOISED_ROW_SUMS.iter().sum();

    // Stochastic rounding is unbiased, so over 200 barcodes the totals agree
    // to well inside a percent. Per-barcode they would not.
    assert_relative_eq!(got_total, want_total, max_relative = 5e-3);
    assert!(
        got_total < fx::RAW_ROW_SUMS.iter().map(|&c| c as f64).sum::<f64>(),
        "denoising removed nothing"
    );
    for (i, &lib) in libs.iter().enumerate() {
        assert!(
            lib <= fx::RAW_ROW_SUMS[i] as f64,
            "barcode {i} gained counts: {lib} > {}",
            fx::RAW_ROW_SUMS[i]
        );
    }

    // Three error terms stack here, none of them the model: f16 storage of the
    // normalised layer, the stochastic rounding of each barcode's library
    // size, and the sub-integer entries that rounded away and so are missing
    // from the layer entirely. See [COL_MARGIN_TOL].
    let col_sums = recover_col_margins(&denoised);
    for (g, (&got, &target)) in col_sums
        .iter()
        .zip(want::DENOISED_COL_SUMS.iter())
        .enumerate()
    {
        assert_relative_eq!(got, target, max_relative = COL_MARGIN_TOL, epsilon = 1e-6);
        assert!(
            got <= fx::RAW_COL_SUMS[g] as f64 + 1e-6,
            "gene {g} gained counts"
        );
    }
}

#[test]
fn test_cellsweep_matches_the_reference_at_the_default_stopping_rule() {
    use fx::default_tol as want;

    let params = CellSweepParams {
        max_iter: want::MAX_ITER,
        ..Default::default()
    };
    let (fit, denoised, _raw, _out) = run_fixture("default", params);

    // Exercises the default stopping rule, which is what a user actually runs.
    // Looser than the fixed-point gate because the parameters have not settled.
    assert_relative_eq!(
        fit.log_likelihood,
        want::LOG_LIKELIHOOD,
        max_relative = DEFAULT_TOL
    );
    // `beta` is still falling towards ~1e-11 when the rule fires, so a
    // one-iteration shift in where it fires moves it by tens of percent.
    // Only check it is in the right neighbourhood; the fixed-point gate holds
    // it tightly.
    assert_relative_eq!(fit.beta, want::BETA, epsilon = 1e-4);

    // Near-zero `alpha` is still drifting when the rule fires: a one-iteration
    // shift moves the 6e-4 barcodes by 2.3e-5, which is 4% relative.
    for (&got, &target) in fit.alpha.iter().zip(want::ALPHA.iter()) {
        assert_relative_eq!(got, target, max_relative = DEFAULT_TOL, epsilon = 1e-4);
    }
    for (&got, &target) in fit.ambient.iter().zip(want::AMBIENT.iter()) {
        assert_relative_eq!(
            got as f64,
            target,
            max_relative = DEFAULT_TOL,
            epsilon = 1e-9
        );
    }
    for (&got, &target) in fit.celltype_profiles.iter().zip(want::PROFILES.iter()) {
        assert_relative_eq!(
            got as f64,
            target,
            max_relative = DEFAULT_TOL,
            epsilon = 1e-9
        );
    }
    assert_eq!(fit.z_hat, want::Z_HAT.to_vec());

    let col_sums = recover_col_margins(&denoised);
    for (&got, &target) in col_sums.iter().zip(want::DENOISED_COL_SUMS.iter()) {
        assert_relative_eq!(got, target, max_relative = 2e-2, epsilon = 1e-6);
    }
}

#[test]
fn test_the_fixture_actually_exercises_denoising() {
    // A parity test against a run that removed nothing would pass while
    // asserting nothing. Flat contamination is genuinely unidentifiable, since
    // adding a constant to every `p_k` explains it equally well, so the
    // fixture skews the ambient profile and varies the per-barcode budget.
    let raw_total: f64 = fx::RAW_ROW_SUMS.iter().map(|&c| c as f64).sum();
    let denoised_total: f64 = fx::tight_tol::DENOISED_ROW_SUMS.iter().sum();
    let kept = denoised_total / raw_total;
    assert!(
        (0.5..0.95).contains(&kept),
        "fixture keeps {kept} of the counts, which is too degenerate to test against"
    );

    // The first block is the ambient-heavy one and has to lose distinctly more
    // than the rest.
    let block = fx::N_GENES / fx::N_CELLTYPES;
    let kept_fraction = |range: std::ops::Range<usize>| {
        let raw: f64 = range.clone().map(|g| fx::RAW_COL_SUMS[g] as f64).sum();
        let denoised: f64 = range
            .map(|g| fx::tight_tol::DENOISED_COL_SUMS[g])
            .sum::<f64>();
        denoised / raw
    };
    assert!(
        kept_fraction(0..block) < kept_fraction(block..fx::N_GENES) - 0.1,
        "the ambient-heavy block was not preferentially stripped"
    );

    // And alpha has to vary, or the per-cell parameter is doing no work.
    let min = fx::tight_tol::ALPHA
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = fx::tight_tol::ALPHA
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(max - min > 0.1, "alpha spans only {min} to {max}");
}
