//! Implementations of the reciprocal best hit (RBH) methods for intersections
//! or correlations, based on the works of Cantini, et al., Bioinformatics,
//! 2019

use faer::{Mat, unzip, zip};
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;

use crate::core::base::cors_similarity::{cor_two_matrices, set_similarity};
use crate::core::mat_struct::NamedMatrix;
use crate::prelude::*;
use crate::utils::vec_utils::{array_max, flatten_vector};

/////////////
// Helpers //
/////////////

/// Identify the k-th largest values
///
/// ### Params
///
/// * `values` - The slice of numerics to look at
/// * `k` - The k-th values to check
///
/// ### Returns
///
/// The k-th largest value
fn kth_largest<T: BixverseFloat>(values: &[T], k: usize) -> T {
    let mut sorted: Vec<T> = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idx = k.saturating_sub(1).min(sorted.len().saturating_sub(1));
    sorted[idx]
}

/// Helper to get the names by index order
///
/// ### Params
///
/// * `names` - The BTreeMap containing string to index information
///
/// ### Returns
///
/// The names by index order
fn names_by_index(names: &BTreeMap<String, usize>) -> Vec<&str> {
    let mut pairs: Vec<(&str, usize)> = names.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    pairs.sort_by_key(|(_, idx)| *idx);
    pairs.into_iter().map(|(k, _)| k).collect()
}

////////////////
// Structures //
////////////////

/// Structure for an Rbh triplet Result
#[derive(Clone, Debug)]
pub struct RbhTripletStruc<'a, T> {
    /// Name of module 1 of the RBH hit
    pub t1: &'a str,
    /// Name of module 2 of the RBH hit
    pub t2: &'a str,
    /// Similarity value between the two
    pub sim: T,
}

/// Structure to store the RBH results.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RbhResult<T> {
    /// Name of the origin data set
    pub origin: String,
    /// Name of the target data set
    pub target: String,
    /// Names of the origin modules/gene sets
    pub origin_modules: Vec<String>,
    /// Names of the target modules/gene sets
    pub target_modules: Vec<String>,
    /// Similarities between the modules/gene sets
    pub similarities: Vec<T>,
}

////////////////////
// Set similarity //
////////////////////

/// Calculates the reciprocal best hits based on set similarities.
///
/// Function will calculate the set similarities (Jaccard or Overlap
/// coefficient) between all of the gene sets between the two data sets and
/// calculate the reciprocal best hits based on this similarity matrix.
/// Option to return reciprocal k-best hits.
///
/// ### Params
///
/// * `origin_modules` - A BTreeMap containing the identified modules of the
///   the origin data set.
/// * `target_modules` - A BTreeMap containing the identified modules of the
///   the target data set.
/// * `k` - Top k hits to consider. If `k=1` this becomes the reciprocal best
///   hit.
/// * `overlap_coefficient` - Shall the overlap coefficient be used instead of
///   Jaccard similarity.
/// * `min_similarity` - Minimum similarity to be returned
///
/// ### Returns
///
/// A vector of `RbhTripletStruc`.
pub fn calculate_rbh_set<'a, T>(
    origin_modules: &'a BTreeMap<String, FxHashSet<String>>,
    target_modules: &'a BTreeMap<String, FxHashSet<String>>,
    k: usize,
    overlap_coefficient: bool,
    min_similarity: T,
) -> Vec<RbhTripletStruc<'a, T>>
where
    T: BixverseFloat,
{
    let names_targets: Vec<&String> = target_modules.keys().collect();
    let names_origin: Vec<&String> = origin_modules.keys().collect();

    let similarities_flat: Vec<Vec<T>> = origin_modules
        .values()
        .map(|v1| {
            target_modules
                .values()
                .map(|v2| set_similarity(v1, v2, overlap_coefficient))
                .collect()
        })
        .collect();

    let mat_data: Vec<T> = flatten_vector(similarities_flat);

    let max_sim = array_max(&mat_data);

    if max_sim < min_similarity {
        vec![RbhTripletStruc {
            t1: "NA",
            t2: "NA",
            sim: T::zero(),
        }]
    } else {
        let nrow = names_origin.len();
        let ncol = names_targets.len();

        let sim_mat = Mat::from_fn(nrow, ncol, |i, j| mat_data[j + i * ncol]);

        let row_thresholds: Vec<T> = sim_mat
            .row_iter()
            .map(|x| {
                let row: Vec<T> = x.iter().copied().collect();
                kth_largest(&row, k)
            })
            .collect();

        let col_thresholds: Vec<T> = sim_mat
            .col_iter()
            .map(|x| {
                let row: Vec<T> = x.iter().copied().collect();
                kth_largest(&row, k)
            })
            .collect();

        let mut matching_pairs: Vec<RbhTripletStruc<T>> = Vec::new();

        for r in 0..nrow {
            for c in 0..ncol {
                let value = sim_mat[(r, c)];
                if value >= row_thresholds[r] && value >= col_thresholds[c] {
                    let triplet = RbhTripletStruc {
                        t1: names_origin[r],
                        t2: names_targets[c],
                        sim: value,
                    };

                    matching_pairs.push(triplet)
                }
            }
        }

        if !matching_pairs.is_empty() {
            matching_pairs
        } else {
            vec![RbhTripletStruc {
                t1: "NA",
                t2: "NA",
                sim: T::zero(),
            }]
        }
    }
}

///////////////////////
// Correlation based //
///////////////////////

/// Calculate the RBH based on correlation of two NamedMatrices
///
/// The function will intersect into shared features and calculate the
/// correlation matrix and subsequently reciprocal best hits based on the
/// absolute correlation. Option to return reciprocal k-best hits.
///
/// ### Params
///
/// * `x1` - `NamedMatrix` of the origin data
/// * `x2` - `NamedMatrix` of the target data
/// * `k` - Top k hits to consider. If `k=1` this becomes the reciprocal best
///   hit.
/// * `spearman` - Shall Spearman correlations be used.
///
/// ### Returns
///
/// A vector of `RbhTripletStruc`.
pub fn calculate_rbh_cor<'a, T>(
    x1: &'a NamedMatrix<'a, T>,
    x2: &'a NamedMatrix<'a, T>,
    k: usize,
    spearman: bool,
) -> Vec<RbhTripletStruc<'a, T>>
where
    T: BixverseFloat,
{
    let row_names_1: FxHashSet<String> = x1.row_names.keys().cloned().collect();
    let row_names_2: FxHashSet<String> = x2.row_names.keys().cloned().collect();

    let names_targets: Vec<&str> = names_by_index(&x1.col_names);
    let names_origin: Vec<&str> = names_by_index(&x2.col_names);

    let intersecting_rows: Vec<String> = row_names_1.intersection(&row_names_2).cloned().collect();

    // early return if there are no intersecting rows
    if intersecting_rows.is_empty() {
        return vec![RbhTripletStruc {
            t1: "NA",
            t2: "NA",
            sim: T::zero(),
        }];
    }

    let row_refs: Vec<&str> = intersecting_rows.iter().map(|s| s.as_str()).collect();
    let sub_x1 = x1.get_rows(&row_refs).unwrap();
    let sub_x2 = x2.get_rows(&row_refs).unwrap();

    let mut correlations = cor_two_matrices(&sub_x1.as_ref(), &sub_x2.as_ref(), spearman);

    zip!(correlations.as_mut()).for_each(|unzip!(x)| *x = x.abs());

    let row_thresholds: Vec<T> = correlations
        .row_iter()
        .map(|x| {
            let row: Vec<T> = x.iter().copied().collect();
            kth_largest(&row, k)
        })
        .collect();

    let col_thresholds: Vec<T> = correlations
        .col_iter()
        .map(|x| {
            let col: Vec<T> = x.iter().copied().collect();
            kth_largest(&col, k)
        })
        .collect();

    let mut matching_pairs: Vec<RbhTripletStruc<'a, T>> = Vec::new();
    let nrow = names_targets.len();
    let ncol = names_origin.len();

    for r in 0..nrow {
        for c in 0..ncol {
            let value = correlations[(r, c)];
            if value >= row_thresholds[r] && value >= col_thresholds[c] {
                let triplet = RbhTripletStruc {
                    t1: names_targets[r],
                    t2: names_origin[c],
                    sim: value,
                };
                matching_pairs.push(triplet);
            }
        }
    }

    if !matching_pairs.is_empty() {
        matching_pairs
    } else {
        vec![RbhTripletStruc {
            t1: "NA",
            t2: "NA",
            sim: T::zero(),
        }]
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /////////
    // Set //
    /////////

    fn module(items: &[&str]) -> FxHashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn modules(pairs: &[(&str, &[&str])]) -> BTreeMap<String, FxHashSet<String>> {
        pairs
            .iter()
            .map(|(n, it)| (n.to_string(), module(it)))
            .collect()
    }

    /// Helper used by the k-best tests below.
    ///
    /// Built so that the resulting Jaccard matrix is:
    /// ```text
    ///        X       Y      Z
    /// A    1.0    0.091   0.0
    /// B   0.333    0.2   0.333
    /// C    0.0    0.333   1.0
    /// ```
    /// At k = 1 only (A,X) and (C,Z) are reciprocal; at k = 2 the set grows
    /// to include (B,X), (B,Z) and (C,Y).
    fn three_by_three_modules() -> (
        BTreeMap<String, FxHashSet<String>>,
        BTreeMap<String, FxHashSet<String>>,
    ) {
        let origin = modules(&[
            ("A", &["1", "2", "3", "4", "5", "6"]),
            ("B", &["4", "5", "6", "7", "8", "9"]),
            ("C", &["7", "8", "9", "10", "11", "12"]),
        ]);
        let target = modules(&[
            ("X", &["1", "2", "3", "4", "5", "6"]),
            ("Y", &["6", "7", "10", "11", "13", "14"]),
            ("Z", &["7", "8", "9", "10", "11", "12"]),
        ]);
        (origin, target)
    }

    fn pair_set<T>(result: &[RbhTripletStruc<T>]) -> FxHashSet<(String, String)> {
        result
            .iter()
            .map(|r| (r.t1.to_string(), r.t2.to_string()))
            .collect()
    }

    /// Perfectly matching module pairs come back as reciprocal best hits at similarity 1.
    #[test]
    fn k1_identifies_clear_reciprocal_hits() {
        let origin = modules(&[("A", &["g1", "g2", "g3"]), ("B", &["g4", "g5", "g6"])]);
        let target = modules(&[("X", &["g1", "g2", "g3"]), ("Y", &["g4", "g5", "g6"])]);

        let result = calculate_rbh_set(&origin, &target, 1, false, 0.0_f64);

        assert_eq!(result.len(), 2);
        let pairs = pair_set(&result);
        assert!(pairs.contains(&("A".into(), "X".into())));
        assert!(pairs.contains(&("B".into(), "Y".into())));
        for r in &result {
            assert!((r.sim - 1.0).abs() < 1e-12);
        }
    }

    /// A hit that is best in its row but not in its column must not survive k = 1.
    #[test]
    fn k1_excludes_non_reciprocal_hits() {
        let (origin, target) = three_by_three_modules();

        let result = calculate_rbh_set(&origin, &target, 1, false, 0.0_f64);

        let pairs = pair_set(&result);
        let expected: FxHashSet<(String, String)> =
            [("A".into(), "X".into()), ("C".into(), "Z".into())]
                .into_iter()
                .collect();
        assert_eq!(pairs, expected);
    }

    /// Raising k to 2 admits the second-best hits that k = 1 discards.
    #[test]
    fn k2_includes_pairs_missed_by_k1() {
        let (origin, target) = three_by_three_modules();

        let result = calculate_rbh_set(&origin, &target, 2, false, 0.0_f64);

        let pairs = pair_set(&result);
        let expected: FxHashSet<(String, String)> = [
            ("A".into(), "X".into()),
            ("B".into(), "X".into()),
            ("B".into(), "Z".into()),
            ("C".into(), "Y".into()),
            ("C".into(), "Z".into()),
        ]
        .into_iter()
        .collect();
        assert_eq!(pairs, expected);
    }

    /// A k at or above both dimensions thresholds nothing, so every pair is returned.
    #[test]
    fn k_at_least_as_large_as_dim_returns_all_pairs() {
        let (origin, target) = three_by_three_modules();

        let result = calculate_rbh_set(&origin, &target, 5, false, 0.0_f64);

        assert_eq!(result.len(), 9);
    }

    /// Nothing clearing `min_similarity` yields the single NA placeholder, not an empty vector.
    #[test]
    fn returns_na_when_max_below_threshold() {
        let origin = modules(&[("A", &["g1", "g2"])]);
        let target = modules(&[("X", &["g3", "g4"])]);

        let result = calculate_rbh_set(&origin, &target, 1, false, 0.5_f64);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].t1, "NA");
        assert_eq!(result[0].t2, "NA");
        assert_eq!(result[0].sim, 0.0);
    }

    /// The overlap coefficient flag switches metric: a strict subset scores 1.0, not 0.25.
    #[test]
    fn overlap_coefficient_differs_from_jaccard() {
        let origin = modules(&[("A", &["g1"])]);
        let target = modules(&[("X", &["g1", "g2", "g3", "g4"])]);

        let jaccard = calculate_rbh_set(&origin, &target, 1, false, 0.0_f64);
        let overlap = calculate_rbh_set(&origin, &target, 1, true, 0.0_f64);

        assert!((jaccard[0].sim - 0.25).abs() < 1e-12);
        assert!((overlap[0].sim - 1.0).abs() < 1e-12);
    }

    //////////////////
    // Correlations //
    //////////////////

    fn mat_from_rows(rows: &[&[f64]]) -> Mat<f64> {
        let nrow = rows.len();
        let ncol = rows[0].len();
        Mat::from_fn(nrow, ncol, |i, j| rows[i][j])
    }

    fn build_named<'a>(
        values: &'a Mat<f64>,
        row_names: &[&str],
        col_names: &[&str],
    ) -> NamedMatrix<'a, f64> {
        NamedMatrix {
            col_names: col_names
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            row_names: row_names
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            values: values.as_ref(),
        }
    }

    /// Two named matrices designed so the Pearson correlation matrix after `abs()` is:
    /// ```text
    ///         X       Y
    /// A     1.000   0.944
    /// B     0.944   1.000
    /// ```
    /// At k = 1 only (A,X) and (B,Y) are reciprocal; at k = 2 every entry passes both
    /// thresholds (the 2nd-largest in each row and column is 0.944), so all 4 pairs return.
    fn two_by_two_cor_setup() -> (Mat<f64>, Mat<f64>) {
        let m1 = mat_from_rows(&[&[1.0, 1.0], &[2.0, 1.0], &[3.0, 2.0], &[4.0, 3.0]]);
        let m2 = m1.clone();
        (m1, m2)
    }

    /// Identical inputs give a diagonal of 1.0, so k = 1 returns exactly the matched column pairs.
    #[test]
    fn cor_k1_identifies_perfect_pairs() {
        let (m1, m2) = two_by_two_cor_setup();
        let x1 = build_named(&m1, &["g1", "g2", "g3", "g4"], &["A", "B"]);
        let x2 = build_named(&m2, &["g1", "g2", "g3", "g4"], &["X", "Y"]);

        let result = calculate_rbh_cor(&x1, &x2, 1, false);

        let pairs: FxHashSet<(String, String)> = result
            .iter()
            .map(|r| (r.t1.to_string(), r.t2.to_string()))
            .collect();
        let expected: FxHashSet<(String, String)> =
            [("A".into(), "X".into()), ("B".into(), "Y".into())]
                .into_iter()
                .collect();
        assert_eq!(pairs, expected);

        for r in &result {
            assert!((r.sim - 1.0).abs() < 1e-10);
        }
    }

    /// At k = 2 every entry of a 2x2 correlation matrix clears both thresholds.
    #[test]
    fn cor_k2_returns_all_pairs_in_2x2() {
        let (m1, m2) = two_by_two_cor_setup();
        let x1 = build_named(&m1, &["g1", "g2", "g3", "g4"], &["A", "B"]);
        let x2 = build_named(&m2, &["g1", "g2", "g3", "g4"], &["X", "Y"]);

        let result = calculate_rbh_cor(&x1, &x2, 2, false);

        assert_eq!(result.len(), 4);
    }

    /// Perfectly anticorrelated columns still pair, since the correlation matrix is taken absolute.
    #[test]
    fn cor_uses_absolute_correlation() {
        // m2 = -m1, so cor(A,X) = cor(B,Y) = -1; after abs they should still pair.
        let m1 = mat_from_rows(&[&[1.0, 1.0], &[2.0, 1.0], &[3.0, 2.0], &[4.0, 3.0]]);
        let m2 = mat_from_rows(&[&[-1.0, -1.0], &[-2.0, -1.0], &[-3.0, -2.0], &[-4.0, -3.0]]);
        let x1 = build_named(&m1, &["g1", "g2", "g3", "g4"], &["A", "B"]);
        let x2 = build_named(&m2, &["g1", "g2", "g3", "g4"], &["X", "Y"]);

        let result = calculate_rbh_cor(&x1, &x2, 1, false);

        let pairs: FxHashSet<(String, String)> = result
            .iter()
            .map(|r| (r.t1.to_string(), r.t2.to_string()))
            .collect();
        assert!(pairs.contains(&("A".into(), "X".into())));
        assert!(pairs.contains(&("B".into(), "Y".into())));

        for r in &result {
            let p = (r.t1, r.t2);
            if p == ("A", "X") || p == ("B", "Y") {
                assert!((r.sim - 1.0).abs() < 1e-10);
            }
        }
    }

    /// Matrices with no shared row names return the NA placeholder rather than correlating nothing.
    #[test]
    fn cor_no_row_intersection_returns_na() {
        let m1 = mat_from_rows(&[&[1.0], &[2.0]]);
        let m2 = mat_from_rows(&[&[1.0], &[2.0]]);
        let x1 = build_named(&m1, &["g1", "g2"], &["A"]);
        let x2 = build_named(&m2, &["g3", "g4"], &["X"]);

        let result = calculate_rbh_cor(&x1, &x2, 1, false);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].t1, "NA");
        assert_eq!(result[0].t2, "NA");
        assert_eq!(result[0].sim, 0.0);
    }

    /// Rows in only one matrix are dropped before correlating, so their values cannot leak in.
    #[test]
    fn cor_uses_only_intersecting_rows() {
        // x1 has g1..g4; x2 has g2..g5. Shared rows are {g2, g3, g4}.
        // On those shared rows: A=[2,3,4], B=[4,2,3], X=A, Y=B.
        // After abs: cor matrix is [[1.0, 0.5],[0.5, 1.0]], so k=1 yields (A,X),(B,Y).
        // g1 and g5 carry junk values that would distort the cors if they leaked in.
        let m1 = mat_from_rows(&[&[1.0, 1.0], &[2.0, 4.0], &[3.0, 2.0], &[4.0, 3.0]]);
        let m2 = mat_from_rows(&[&[2.0, 4.0], &[3.0, 2.0], &[4.0, 3.0], &[99.0, -99.0]]);
        let x1 = build_named(&m1, &["g1", "g2", "g3", "g4"], &["A", "B"]);
        let x2 = build_named(&m2, &["g2", "g3", "g4", "g5"], &["X", "Y"]);

        let result = calculate_rbh_cor(&x1, &x2, 1, false);

        let pairs: FxHashSet<(String, String)> = result
            .iter()
            .map(|r| (r.t1.to_string(), r.t2.to_string()))
            .collect();
        let expected: FxHashSet<(String, String)> =
            [("A".into(), "X".into()), ("B".into(), "Y".into())]
                .into_iter()
                .collect();
        assert_eq!(pairs, expected);
    }

    /// Columns are resolved by stored index, not by the BTreeMap's alphabetical key order.
    #[test]
    fn cor_respects_non_alphabetical_column_order() {
        let m1 = mat_from_rows(&[&[1.0, 1.0], &[2.0, 1.0], &[3.0, 2.0], &[4.0, 3.0]]);
        let m2 = mat_from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[2.0, 3.0], &[3.0, 4.0]]);

        let x1 = NamedMatrix {
            col_names: [("Z".to_string(), 0), ("A".to_string(), 1)]
                .into_iter()
                .collect(),
            row_names: ["g1", "g2", "g3", "g4"]
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            values: m1.as_ref(),
        };
        let x2 = NamedMatrix {
            col_names: [("A".to_string(), 0), ("Z".to_string(), 1)]
                .into_iter()
                .collect(),
            row_names: ["g1", "g2", "g3", "g4"]
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            values: m2.as_ref(),
        };

        let result = calculate_rbh_cor(&x1, &x2, 1, false);

        // correct alignment yields Z<->Z and A<->A, both at correlation 1.0.
        let pairs: FxHashSet<(String, String)> = result
            .iter()
            .map(|r| (r.t1.to_string(), r.t2.to_string()))
            .collect();
        assert!(pairs.contains(&("Z".into(), "Z".into())));
        assert!(pairs.contains(&("A".into(), "A".into())));
        for r in &result {
            assert!((r.sim - 1.0).abs() < 1e-10);
        }
    }
}
