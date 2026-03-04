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
/// Function will calculate the set similarities (Jaccard or Overlap coefficient)
/// between all of the gene sets between the two data sets and calculate the
/// reciprocal best hits based on this similarity matrix.
///
/// ### Params
///
/// * `origin_modules` - A BTreeMap containing the identified modules of the
///   the origin data set.
/// * `target_modules` - A BTreeMap containing the identified modules of the
///   the target data set.
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

        let row_maxima: Vec<&T> = sim_mat
            .row_iter()
            .map(|x| {
                let row: Vec<&T> = x.iter().collect();
                array_max(&row)
            })
            .collect();

        let col_maxima: Vec<&T> = sim_mat
            .col_iter()
            .map(|x| {
                let col: Vec<&T> = x.iter().collect();
                array_max(&col)
            })
            .collect();

        let mut matching_pairs: Vec<RbhTripletStruc<T>> = Vec::new();

        for r in 0..nrow {
            for c in 0..ncol {
                let value = sim_mat[(r, c)];
                if value == *row_maxima[r] && value == *col_maxima[c] {
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
/// The function will intersect into shared features and calculate the correlation
/// matrix and subsequently reciprocal best hits based on the absolute correlation.
///
/// ### Params
///
/// * `x1` - `NamedMatrix` of the origin data
/// * `x2` - `NamedMatrix` of the target data
/// * `spearman` - Shall Spearman correlations be used.
///
/// ### Returns
///
/// A vector of `RbhTripletStruc`.
pub fn calculate_rbh_cor<'a, T>(
    x1: &'a NamedMatrix<'a, T>,
    x2: &'a NamedMatrix<'a, T>,
    spearman: bool,
) -> Vec<RbhTripletStruc<'a, T>>
where
    T: BixverseFloat,
{
    let row_names_1: FxHashSet<String> = x1.row_names.keys().cloned().collect();
    let row_names_2: FxHashSet<String> = x2.row_names.keys().cloned().collect();

    // Now these references will live as long as 'a because they reference x1 and x2
    let names_targets: Vec<&str> = x1.col_names.keys().map(|s| s.as_str()).collect();
    let names_origin: Vec<&str> = x2.col_names.keys().map(|s| s.as_str()).collect();

    let intersecting_rows: Vec<String> = row_names_1.intersection(&row_names_2).cloned().collect();

    // Early return if there are no intersecting rows
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

    let row_maxima: Vec<&T> = correlations
        .row_iter()
        .map(|x| {
            let row: Vec<&T> = x.iter().collect();
            array_max(&row)
        })
        .collect();

    let col_maxima: Vec<&T> = correlations
        .col_iter()
        .map(|x| {
            let col: Vec<&T> = x.iter().collect();
            array_max(&col)
        })
        .collect();

    let mut matching_pairs: Vec<RbhTripletStruc<'a, T>> = Vec::new();
    let nrow = names_targets.len();
    let ncol = names_origin.len();

    for r in 0..nrow {
        for c in 0..ncol {
            let value = correlations[(r, c)];
            if value == *row_maxima[r] && value == *col_maxima[c] {
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
