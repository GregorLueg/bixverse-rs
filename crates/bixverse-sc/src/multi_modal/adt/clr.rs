//! Centred log-ratio (CLR) normalisation of ADT counts, one cell at a time.

use bixverse_core::utils::traits::BixverseFloat;
use faer::{Mat, MatRef};
use rayon::prelude::*;

/// CLR-normalise ADT counts per cell (row)
///
/// Two variants:
///
/// - proper CLR: `log1p(x) - mean(log1p(x))`, mean-centred, can be negative.
/// - Seurat-style: `log1p(x / g)` with `g = exp(sum(log1p(x[x > 0])) / ncol)`,
///   non-negative.
///
/// The per-cell scale is computed in parallel over the rows; the row sums run
/// in column order, so the result does not depend on the thread count.
///
/// ### Params
///
/// * `counts` - Cells x features raw counts
/// * `seurat_clr` - Use the Seurat variant instead of the proper CLR
///
/// ### Returns
///
/// Cells x features CLR-normalised counts.
pub fn adt_clr<T>(counts: MatRef<T>, seurat_clr: bool) -> Mat<T>
where
    T: BixverseFloat + Send + Sync,
{
    let (nrow, ncol) = (counts.nrows(), counts.ncols());
    let n = T::from_usize(ncol).expect("feature count fits the float type");

    let scale: Vec<T> = (0..nrow)
        .into_par_iter()
        .map(|i| {
            let mut sum = T::zero();
            for j in 0..ncol {
                let v = counts[(i, j)];
                if !seurat_clr || v > T::zero() {
                    sum += v.ln_1p();
                }
            }
            if seurat_clr { (sum / n).exp() } else { sum / n }
        })
        .collect();

    Mat::from_fn(nrow, ncol, |i, j| {
        let v = counts[(i, j)];
        if seurat_clr {
            (v / scale[i]).ln_1p()
        } else {
            v.ln_1p() - scale[i]
        }
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The proper CLR centres every row on zero.
    #[test]
    fn test_adt_clr_rows_centred() {
        let counts = Mat::from_fn(4, 5, |i, j| ((i + 1) * (j + 2)) as f64);
        let res = adt_clr(counts.as_ref(), false);
        for i in 0..4 {
            let s: f64 = (0..5).map(|j| res[(i, j)]).sum();
            assert_relative_eq!(s, 0.0, epsilon = 1e-12);
        }
        // and one closed-form value, so the log and pseudocount are pinned:
        // row 0 is 2..=6, centred on the mean of their log1p
        let mean = (2..=6).map(|x| (x as f64).ln_1p()).sum::<f64>() / 5.0;
        assert_relative_eq!(res[(0, 0)], 2.0f64.ln_1p() - mean, epsilon = 1e-12);
    }

    /// The Seurat variant matches its closed form and stays non-negative,
    /// zeros included.
    #[test]
    fn test_adt_clr_seurat_known_values() {
        let counts = Mat::from_fn(1, 3, |_, j| [0.0, 3.0, 7.0][j]);
        let res = adt_clr(counts.as_ref(), true);
        let g = ((3.0f64.ln_1p() + 7.0f64.ln_1p()) / 3.0).exp();
        assert_relative_eq!(res[(0, 0)], 0.0);
        assert_relative_eq!(res[(0, 1)], (3.0 / g).ln_1p(), epsilon = 1e-12);
        assert_relative_eq!(res[(0, 2)], (7.0 / g).ln_1p(), epsilon = 1e-12);
    }
}
