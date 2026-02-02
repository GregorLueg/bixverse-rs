use faer::{ColRef, Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::BixverseFloat;

////////////////
// Infotheory //
////////////////

/// Calculate the mutual information between two column references
///
/// The columns need to be binned
///
/// ### Params
///
/// * `col_i` - Column reference to the first column to compare
/// * `col_j` - Column reference to the second column to compare
/// * `n_bins` - Optional number of bins. If not provided, will default to
///   `sqrt(nrows)`.
///
/// ### Returns
///
/// The mutual information between the two columns
pub fn calculate_mi<T>(col_i: ColRef<usize>, col_j: ColRef<usize>, n_bins: Option<usize>) -> T
where
    T: BixverseFloat,
{
    let n_rows = col_i.nrows();
    let n_bins =
        n_bins.unwrap_or_else(|| T::from_usize(n_rows).unwrap().sqrt().to_usize().unwrap());
    let mut joint_counts = vec![vec![0usize; n_bins]; n_bins];
    let mut marginal_i = vec![0usize; n_bins];
    let mut marginal_j = vec![0usize; n_bins];
    for i in 0..n_rows {
        let bin_i = col_i[i];
        let bin_j = col_j[i];
        joint_counts[bin_i][bin_j] += 1;
        marginal_i[bin_i] += 1;
        marginal_j[bin_j] += 1;
    }
    let n = T::from_usize(n_rows).unwrap();
    let mut mi = T::zero();
    for i in 0..n_bins {
        for j in 0..n_bins {
            let joint_prob = T::from_usize(joint_counts[i][j]).unwrap() / n;
            if joint_prob > T::zero() {
                let marginal_prob_i = T::from_usize(marginal_i[i]).unwrap() / n;
                let marginal_prob_j = T::from_usize(marginal_j[j]).unwrap() / n;
                mi += joint_prob * (joint_prob / (marginal_prob_i * marginal_prob_j)).ln();
            }
        }
    }
    mi
}

/// Calculates the joint entropy between two column references
///
/// ### Params
///
/// * `col_i` - Column reference to the first column to compare
/// * `col_j` - Column reference to the second column to compare
/// * `n_bins` - Optional number of bins. If not provided, will default to
///   `sqrt(nrows)`.
///
/// ### Returns
///
/// The joint entropy
pub fn calculate_joint_entropy<T>(
    col_i: ColRef<usize>,
    col_j: ColRef<usize>,
    n_bins: Option<usize>,
) -> T
where
    T: BixverseFloat,
{
    let n_rows = col_i.nrows();
    let n_bins =
        n_bins.unwrap_or_else(|| T::from_usize(n_rows).unwrap().sqrt().to_usize().unwrap());
    let mut joint_counts = vec![vec![0usize; n_bins]; n_bins];
    for i in 0..n_rows {
        joint_counts[col_i[i]][col_j[i]] += 1;
    }
    let n = T::from_usize(n_rows).unwrap();
    let mut joint_entropy = T::zero();
    for i in 0..n_bins {
        for j in 0..n_bins {
            let joint_prob = T::from_usize(joint_counts[i][j]).unwrap() / n;
            if joint_prob > T::zero() {
                joint_entropy -= joint_prob * joint_prob.ln();
            }
        }
    }
    joint_entropy
}

/// Calculates the entropy of a column reference
///
/// The column needs to be binned
///
/// ### Params
///
/// * `col` - Column reference for which to calculate entropy
/// * `n_bins` - Optional number of bins. If not provided, will default to
///   `sqrt(nrows)`.
///
/// ### Returns
///
/// The entropy of the column
pub fn calculate_entropy<T>(col: ColRef<usize>, n_bins: Option<usize>) -> T
where
    T: BixverseFloat,
{
    let n_rows = col.nrows();
    let n_bins =
        n_bins.unwrap_or_else(|| T::from_usize(n_rows).unwrap().sqrt().to_usize().unwrap());
    let mut counts = vec![0usize; n_bins];
    for i in 0..n_rows {
        counts[col[i]] += 1;
    }
    let n = T::from_usize(n_rows).unwrap();
    let mut entropy = T::zero();
    for &count in &counts {
        if count > 0 {
            let prob = T::from_usize(count).unwrap() / n;
            entropy -= prob * prob.ln();
        }
    }
    entropy
}

/////////////
// Binning //
/////////////

/// Binning strategy enum
#[derive(Debug, Clone, Default)]
pub enum BinningStrategy {
    /// Equal width bins (equal distance between bin edges)
    #[default]
    EqualWidth,
    /// Equal frequency bins (approximately equal number of values per bin)
    EqualFrequency,
}

/// Parsing the binning strategy
///
/// ### Params
///
/// * `s` - string defining the binning strategy
///
/// ### Returns
///
/// The `BinningStrategy`.
pub fn parse_bin_strategy_type(s: &str) -> Option<BinningStrategy> {
    match s.to_lowercase().as_str() {
        "equal_width" => Some(BinningStrategy::EqualWidth),
        "equal_freq" => Some(BinningStrategy::EqualFrequency),
        _ => None,
    }
}

/// Equal width binning for a single column
///
/// ### Params
///
/// * `col` - Column refence to bin with equal width strategy
/// * `n_bins` - Number of bins to use
///
/// ### Returns
///
/// Binned vector
fn bin_equal_width<T>(col: &ColRef<T>, n_bins: usize) -> Vec<usize>
where
    T: BixverseFloat,
{
    let (min_val, max_val) = col.iter().fold(
        (
            T::from_f64(f64::INFINITY).unwrap(),
            T::from_f64(f64::NEG_INFINITY).unwrap(),
        ),
        |(min, max), x| (min.min(*x), max.max(*x)),
    );
    let range = max_val - min_val;
    if range == T::zero() {
        return vec![0; col.nrows()];
    }
    let step = range / T::from_usize(n_bins).unwrap();
    col.iter()
        .map(|x| {
            let bin = ((*x - min_val) / step).floor().to_usize().unwrap();
            bin.min(n_bins - 1)
        })
        .collect()
}

/// Equal frequency binning for a single column
///
/// ### Params
///
/// * `col` - Column refence to bin with equal frequency strategy
/// * `n_bins` - Number of bins to use
///
/// ### Returns
///
/// Binned vector
fn bin_equal_frequency<T>(col: &ColRef<T>, n_bins: usize) -> Vec<usize>
where
    T: BixverseFloat,
{
    let n_rows = col.nrows();
    let mut sorted_values: Vec<T> = col.iter().copied().collect();
    sorted_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut split_points = Vec::with_capacity(n_bins);
    for k in 1..n_bins {
        let idx = (k * n_rows) / n_bins;
        split_points.push(sorted_values[idx.min(n_rows - 1)]);
    }

    let mut bin_assignments = vec![0; n_rows];
    for (i, &value) in col.iter().enumerate() {
        let mut bin = 0;
        for &split in &split_points {
            if value >= split {
                bin += 1;
            } else {
                break;
            }
        }
        bin_assignments[i] = bin.min(n_bins - 1);
    }
    bin_assignments
}

/// Column wise binning
///
/// ### Params
///
/// * `mat` - The matrix on which to apply column-wise binning
/// * `n_bins` - Optional number of bins. If not provided, will default to
///   `sqrt(nrow)`
///
/// ### Returns
///
/// The matrix with the columns being binned into equal distances.
pub fn bin_matrix_cols<T>(
    mat: &MatRef<T>,
    n_bins: Option<usize>,
    strategy: BinningStrategy,
) -> Mat<usize>
where
    T: BixverseFloat,
{
    let (n_rows, n_cols) = mat.shape();
    let n_bins =
        n_bins.unwrap_or_else(|| T::from_usize(n_rows).unwrap().sqrt().to_usize().unwrap());
    let binned_vals: Vec<Vec<usize>> = mat
        .par_col_iter()
        .map(|col| match strategy {
            BinningStrategy::EqualWidth => bin_equal_width(&col, n_bins),
            BinningStrategy::EqualFrequency => bin_equal_frequency(&col, n_bins),
        })
        .collect();
    Mat::from_fn(n_rows, n_cols, |i, j| binned_vals[j][i])
}

#[cfg(test)]
mod tests {
    // Tests focus mainly on API; the Rest was heavily tested within R

    use super::*;
    use faer::mat;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{} != {}", a, b);
    }

    #[test]
    fn test_calculate_entropy_uniform() {
        // Data: [0, 1, 0, 1]
        // 2 bins. P(0) = 0.5, P(1) = 0.5
        // Entropy = -0.5 * ln(0.5) - 0.5 * ln(0.5) = -ln(0.5) = ln(2)
        let data = mat![[0], [1], [0], [1]]; // usize matrix
        let col = data.col(0);

        // Explicitly set 2 bins
        let ent: f64 = calculate_entropy(col, Some(2));
        assert_approx_eq(ent, 2.0_f64.ln());
    }

    #[test]
    fn test_calculate_entropy_zero() {
        // Data: [0, 0, 0, 0]
        // P(0) = 1.0
        // Entropy = -1 * ln(1) = 0
        let data = mat![[0], [0], [0], [0]];
        let col = data.col(0);

        let ent: f64 = calculate_entropy(col, Some(1));
        assert_approx_eq(ent, 0.0);
    }

    #[test]
    fn test_calculate_joint_entropy() {
        // X: [0, 1, 0, 1]
        // Y: [0, 1, 0, 1]
        // Perfectly correlated. Joint entropy H(X,Y) should equal H(X) = ln(2)
        let data = mat![[0, 0], [1, 1], [0, 0], [1, 1]]; // usize matrix

        let je: f64 = calculate_joint_entropy(data.col(0), data.col(1), Some(2));
        assert_approx_eq(je, 2.0_f64.ln());

        // Independent variables
        // X: [0, 0, 1, 1]
        // Y: [0, 1, 0, 1]
        // Joint distribution is uniform over (0,0), (0,1), (1,0), (1,1) -> p=0.25
        // H(X,Y) = -4 * (0.25 * ln(0.25)) = -ln(0.25) = ln(4)
        let data_indep = mat![[0, 0], [0, 1], [1, 0], [1, 1]];
        let je_indep: f64 = calculate_joint_entropy(data_indep.col(0), data_indep.col(1), Some(2));
        assert_approx_eq(je_indep, 4.0_f64.ln());
    }

    #[test]
    fn test_calculate_mi() {
        // X: [0, 1], Y: [0, 1] -> Perfectly correlated
        // MI(X;Y) = H(X) + H(Y) - H(X,Y)
        // H(X)=ln2, H(Y)=ln2, H(X,Y)=ln2
        // MI = ln2
        let data = mat![[0, 0], [1, 1]];
        let mi: f64 = calculate_mi(data.col(0), data.col(1), Some(2));
        assert_approx_eq(mi, 2.0_f64.ln());

        // Independent -> MI should be 0
        // X: [0, 0, 1, 1], Y: [0, 1, 0, 1]
        let data_indep = mat![[0, 0], [0, 1], [1, 0], [1, 1]];
        let mi_indep: f64 = calculate_mi(data_indep.col(0), data_indep.col(1), Some(2));
        assert_approx_eq(mi_indep, 0.0);
    }

    #[test]
    fn test_bin_equal_width() {
        // Data: 0.0, 5.0, 10.0
        // 3 bins. Range=10. Step=3.33...
        // 0.0 -> bin 0
        // 5.0 -> 1.5 -> bin 1
        // 10.0 -> 3.0 -> clamped to bin 2
        let mat = mat![[0.0], [5.0], [10.0]];

        let binned = bin_matrix_cols(&mat.as_ref(), Some(3), BinningStrategy::EqualWidth);

        assert_eq!(binned.nrows(), 3);
        assert_eq!(binned.get(0, 0), &0);
        assert_eq!(binned.get(1, 0), &1);
        assert_eq!(binned.get(2, 0), &2);
    }

    #[test]
    fn test_bin_equal_width_constant_column() {
        // Edge case: Range is 0
        let mat = mat![[1.0], [1.0], [1.0]];
        let binned = bin_matrix_cols(&mat.as_ref(), Some(3), BinningStrategy::EqualWidth);

        // Should default to bin 0 for all
        assert_eq!(binned.get(0, 0), &0);
        assert_eq!(binned.get(1, 0), &0);
        assert_eq!(binned.get(2, 0), &0);
    }

    #[test]
    fn test_bin_equal_frequency() {
        // Data: [1, 2, 8, 9]
        // 2 bins.
        // Sorted: 1, 2 | 8, 9
        // Split should occur roughly between 2 and 8
        let mat = mat![[1.0], [9.0], [2.0], [8.0]];

        let binned = bin_matrix_cols(&mat.as_ref(), Some(2), BinningStrategy::EqualFrequency);

        // 1.0 and 2.0 should be in bin 0
        // 8.0 and 9.0 should be in bin 1
        // Indices in input: 0->1.0, 1->9.0, 2->2.0, 3->8.0
        assert_eq!(binned.get(0, 0), &0); // 1.0
        assert_eq!(binned.get(1, 0), &1); // 9.0
        assert_eq!(binned.get(2, 0), &0); // 2.0
        assert_eq!(binned.get(3, 0), &1); // 8.0
    }

    #[test]
    fn test_parse_bin_strategy() {
        assert!(matches!(
            parse_bin_strategy_type("equal_width"),
            Some(BinningStrategy::EqualWidth)
        ));
        assert!(matches!(
            parse_bin_strategy_type("equal_freq"),
            Some(BinningStrategy::EqualFrequency)
        ));
        assert!(parse_bin_strategy_type("random").is_none());
    }
}
