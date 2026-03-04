//! Implementation of the CoReMo workflow from François, et al., Nat Commun,
//! 2024

use faer::MatRef;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

use crate::prelude::*;

///////////////////////
// Cluster stability //
///////////////////////

/// Helper functions to calculate the intersection of sorted usize vectors
///
/// ### Params
///
/// * `a` - (Sorted) slice of usize
/// * `b` - (Sorted) slice of usize
///
/// ### Returns
///
/// Intersection between the two sorted slices.
pub fn intersection_size_sorted(a: &[usize], b: &[usize]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }

    count
}

/// Function that assesses the cluster stability.
///
/// ### Params
///
/// * `cluster_matrix` - A matrix with the columns representing the bootstraps,
///   the rows the features and the values which cluster the feature belongs to.
///
/// ### Returns
///
/// Returns tuple of (average Jaccard similarities, standard deviations of the
/// Jaccard similarities).
pub fn cluster_stability<T>(cluster_matrix: &MatRef<i32>) -> Vec<(T, T)>
where
    T: BixverseFloat,
{
    let n_features = cluster_matrix.nrows();
    let n_iter = cluster_matrix.ncols();

    // Pre-compute cluster membership maps for all bootstraps
    let bootstrap_cluster_maps: Vec<FxHashMap<i32, Vec<usize>>> = (0..n_iter)
        .into_par_iter()
        .map(|boot_idx| {
            let mut clusters_map: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
            for feature_idx in 0..n_features {
                let cluster_id = cluster_matrix[(feature_idx, boot_idx)];
                clusters_map
                    .entry(cluster_id)
                    .or_default()
                    .push(feature_idx);
            }
            clusters_map
        })
        .collect();

    // Process features in parallel with optimized memory usage
    (0..n_features)
        .into_par_iter()
        .map(|feature_idx| {
            let n_pairs = (n_iter * (n_iter - 1)) / 2;
            let mut jaccard_scores = Vec::with_capacity(n_pairs);

            for i in 0..(n_iter - 1) {
                for j in (i + 1)..n_iter {
                    let cluster_i = cluster_matrix[(feature_idx, i)];
                    let cluster_j = cluster_matrix[(feature_idx, j)];

                    let members_i = bootstrap_cluster_maps[i]
                        .get(&cluster_i)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);

                    let members_j = bootstrap_cluster_maps[j]
                        .get(&cluster_j)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);

                    let intersection_size = intersection_size_sorted(members_i, members_j);
                    let union_size = members_i.len() + members_j.len() - intersection_size;

                    let jaccard = if union_size == 0 {
                        0.0
                    } else {
                        intersection_size as f64 / union_size as f64
                    };
                    jaccard_scores.push(jaccard);
                }
            }

            let mean_jaccard = jaccard_scores.iter().sum::<f64>() / jaccard_scores.len() as f64;
            let variance = jaccard_scores
                .iter()
                .map(|x| (x - mean_jaccard).powi(2))
                .sum::<f64>()
                / jaccard_scores.len() as f64;
            let std_jaccard = variance.sqrt();

            (
                T::from_f64(mean_jaccard).unwrap(),
                T::from_f64(std_jaccard).unwrap(),
            )
        })
        .collect()
}

/// Helper to split correlation matrices
///
/// Function will check the signs of the correlation matrix and return a vector
/// of only `1` if everything has a positive sign, `-1` if everything has a
/// negative sign and a mix if the correlation signs differ. Under the hood it
/// uses a graph-based approach which is fast.
///
/// ### Params
///
/// * `cor_mat` - The correlation matrix to split
///
/// ### Returns
///
/// A vector containing -1 and/or 1, pending input correlation matrix.
pub fn split_cor_mat_by_sign<T>(cor_mat: &MatRef<T>) -> Vec<i32>
where
    T: BixverseFloat,
{
    // assert_symmetric_mat!(cor_mat);

    let n = cor_mat.ncols();
    let total_elem = n * (n - 1) / 2;
    let mut off_diag = Vec::with_capacity(total_elem);

    for i in 0..n {
        for j in (i + 1)..n {
            off_diag.push(*cor_mat.get(i, j));
        }
    }

    let all_pos = off_diag.iter().all(|x| *x >= T::zero());
    let all_neg = off_diag.iter().all(|x| *x <= T::zero());

    if all_pos {
        return vec![1; n];
    }
    if all_neg {
        return vec![-1; n];
    }

    let mut visited = vec![false; n];
    let mut groups = Vec::new();

    for i in 0..n {
        if !visited[i] {
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(i);
            visited[i] = true;

            while let Some(node) = queue.pop_front() {
                component.push(node);

                for j in 0..n {
                    if !visited[j] && *cor_mat.get(node, j) > T::zero() {
                        visited[j] = true;
                        queue.push_back(j);
                    }
                }
            }
            groups.push(component);
        }
    }

    let mut result = vec![-1; n];
    for group in groups {
        // check if this group correlates positively with feature 0
        let has_positive_with_first =
            group.contains(&0) || group.iter().any(|&idx| *cor_mat.get(0, idx) > T::zero());

        let label = if has_positive_with_first { 1 } else { -1 };
        for &idx in &group {
            result[idx] = label;
        }
    }

    result
}
