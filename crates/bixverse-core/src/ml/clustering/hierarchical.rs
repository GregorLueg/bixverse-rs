//! Agglomerative hierarchical clustering on a condensed distance matrix.
//!
//! A port of `scipy.cluster.hierarchy.linkage` for the methods SciPy runs
//! through its nearest-neighbour chain (`ward`, `complete`, `average`,
//! `weighted`), including its tie-breaking, the stable sort of the merges by
//! distance and the relabelling of clusters, so the output matches SciPy's
//! linkage matrix row for row.

use rayon::prelude::*;

use crate::prelude::*;

///////////
// Types //
///////////

/// Linkage criterion, all reducible so the nearest-neighbour chain is exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinkageMethod {
    /// Ward's minimum variance criterion. Expects Euclidean input distances.
    #[default]
    Ward,
    /// Maximum distance between members.
    Complete,
    /// Mean distance between members (UPGMA).
    Average,
    /// Mean of the two merged clusters' distances (WPGMA).
    Weighted,
}

/// Parse a linkage method.
///
/// ### Params
///
/// * `s` - `"ward"`, `"complete"`, `"average"` or `"weighted"`, case-insensitive.
///
/// ### Returns
///
/// The method, or `None` if unrecognised.
pub fn parse_linkage_method(s: &str) -> Option<LinkageMethod> {
    match s.to_lowercase().as_str() {
        "ward" => Some(LinkageMethod::Ward),
        "complete" => Some(LinkageMethod::Complete),
        "average" => Some(LinkageMethod::Average),
        "weighted" => Some(LinkageMethod::Weighted),
        _ => None,
    }
}

/// One merge, a row of SciPy's linkage matrix.
///
/// Leaves are `0..n`; the cluster formed by merge `k` is `n + k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkageStep<T> {
    /// Smaller cluster id of the pair.
    pub left: usize,
    /// Larger cluster id of the pair.
    pub right: usize,
    /// Distance between the two clusters at the merge.
    pub distance: T,
    /// Leaves in the merged cluster.
    pub size: usize,
}

/////////////
// Helpers //
/////////////

/// Position of the pair `(i, j)`, `i != j`, in a condensed `n × n` matrix.
///
/// ### Params
///
/// * `n` - Observations.
/// * `i` - First observation.
/// * `j` - Second observation.
///
/// ### Returns
///
/// The index into the condensed vector.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    n * a - a * (a + 1) / 2 + (b - a - 1)
}

/// Lance-Williams update: the distance from cluster `i` to the merge of `x`
/// and `y`, as SciPy's `_hierarchy_distance_update.pxi`.
///
/// ### Params
///
/// * `method` - Linkage criterion.
/// * `d_xi` - Distance from `x` to `i`.
/// * `d_yi` - Distance from `y` to `i`.
/// * `d_xy` - Distance from `x` to `y`.
/// * `size_x` - Leaves in `x`.
/// * `size_y` - Leaves in `y`.
/// * `size_i` - Leaves in `i`.
///
/// ### Returns
///
/// The updated distance.
#[inline]
fn update_distance<T: BixverseFloat>(
    method: LinkageMethod,
    d_xi: T,
    d_yi: T,
    d_xy: T,
    size_x: usize,
    size_y: usize,
    size_i: usize,
) -> T {
    let sx = T::from_usize(size_x).unwrap();
    let sy = T::from_usize(size_y).unwrap();
    let si = T::from_usize(size_i).unwrap();
    match method {
        LinkageMethod::Ward => {
            let t = T::one() / (sx + sy + si);
            ((si + sx) * t * d_xi * d_xi + (si + sy) * t * d_yi * d_yi - si * t * d_xy * d_xy)
                .sqrt()
        }
        LinkageMethod::Complete => d_xi.max(d_yi),
        LinkageMethod::Average => (sx * d_xi + sy * d_yi) / (sx + sy),
        LinkageMethod::Weighted => T::from_f64(0.5).unwrap() * (d_xi + d_yi),
    }
}

/// Find a root in the union-find, compressing the path.
///
/// ### Params
///
/// * `parent` - Parent per node; updated in place.
/// * `x` - Node.
///
/// ### Returns
///
/// The root of `x`.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    let mut p = x;
    while parent[x] != x {
        x = parent[x];
    }
    while parent[p] != x {
        let next = parent[p];
        parent[p] = x;
        p = next;
    }
    x
}

/// SciPy's `label`: turn merges recorded by surviving slot into cluster ids
/// (`n + k` for merge `k` in sorted order), smaller id first.
///
/// ### Params
///
/// * `steps` - Merges sorted by distance; relabelled in place.
/// * `n` - Observations.
fn relabel<T: Copy>(steps: &mut [LinkageStep<T>], n: usize) {
    let mut parent: Vec<usize> = (0..2 * n - 1).collect();
    let mut size = vec![1usize; 2 * n - 1];
    for (k, step) in steps.iter_mut().enumerate() {
        let x = find_root(&mut parent, step.left);
        let y = find_root(&mut parent, step.right);
        (step.left, step.right) = if x < y { (x, y) } else { (y, x) };
        let label = n + k;
        parent[x] = label;
        parent[y] = label;
        size[label] = size[x] + size[y];
        step.size = size[label];
    }
}

//////////
// Main //
//////////

/// Pairwise Euclidean distances between rows, SciPy's `pdist` layout.
///
/// ### Params
///
/// * `data` - Row-major observations, `n × dim`.
/// * `dim` - Features per observation.
///
/// ### Returns
///
/// The condensed distances: pairs `(i, j)`, `i < j`, in row order, length
/// `n (n - 1) / 2`.
pub fn condensed_euclidean<T: BixverseFloat + Send + Sync>(data: &[T], dim: usize) -> Vec<T> {
    let n = data.len().checked_div(dim).unwrap_or(0);
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let a = &data[i * dim..(i + 1) * dim];
            (i + 1..n).map(move |j| {
                let b = &data[j * dim..(j + 1) * dim];
                a.iter()
                    .zip(b)
                    .fold(T::zero(), |acc, (&u, &v)| acc + (u - v) * (u - v))
                    .sqrt()
            })
        })
        .collect()
}

/// Agglomerative clustering via the nearest-neighbour chain, SciPy's
/// `linkage(y, method)` for the reducible methods.
///
/// The chain prefers its previous element on ties and otherwise the lowest
/// index, the merged cluster takes the higher slot, and the merges are then
/// stably sorted by distance and relabelled, all as SciPy, so ties resolve the
/// same way.
///
/// ### Params
///
/// * `condensed` - Condensed distances, as [`condensed_euclidean`] returns.
/// * `n` - Observations, at least 2.
/// * `method` - Linkage criterion.
///
/// ### Returns
///
/// The `n - 1` merges in ascending distance.
///
/// ### Errors
///
/// `InvalidArgument` for fewer than two observations, `LengthMismatch` if
/// `condensed` is not `n (n - 1) / 2` long, `LinkageNonFiniteDistance` for a
/// NaN or infinite distance.
pub fn linkage<T: BixverseFloat>(
    condensed: &[T],
    n: usize,
    method: LinkageMethod,
) -> Result<Vec<LinkageStep<T>>, CoreErrors> {
    if n < 2 {
        return Err(CoreErrors::InvalidArgument(format!(
            "linkage needs at least two observations, got {n}"
        )));
    }
    let expected = n * (n - 1) / 2;
    if condensed.len() != expected {
        return Err(CoreErrors::LengthMismatch {
            name: "condensed",
            expected,
            found: condensed.len(),
        });
    }
    if let Some(index) = condensed.iter().position(|d| !d.is_finite()) {
        return Err(CoreErrors::LinkageNonFiniteDistance { index });
    }

    let mut d = condensed.to_vec();
    let mut size = vec![1usize; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut steps: Vec<LinkageStep<T>> = Vec::with_capacity(n - 1);

    for _ in 0..n - 1 {
        if chain.is_empty() {
            chain.push(size.iter().position(|&s| s > 0).expect("a live cluster"));
        }

        let (x, y, current_min) = loop {
            let x = chain[chain.len() - 1];
            let (mut y, mut current_min) = if chain.len() > 1 {
                let prev = chain[chain.len() - 2];
                (prev, d[condensed_index(n, x, prev)])
            } else {
                (0, T::infinity())
            };
            for i in 0..n {
                if size[i] == 0 || i == x {
                    continue;
                }
                let dist = d[condensed_index(n, x, i)];
                if dist < current_min {
                    current_min = dist;
                    y = i;
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                break (x, y, current_min);
            }
            chain.push(y);
        };
        chain.truncate(chain.len() - 2);

        let (x, y) = if x > y { (y, x) } else { (x, y) };
        let (nx, ny) = (size[x], size[y]);
        steps.push(LinkageStep {
            left: x,
            right: y,
            distance: current_min,
            size: nx + ny,
        });
        size[x] = 0;
        size[y] = nx + ny;

        for i in 0..n {
            let ni = size[i];
            if ni == 0 || i == y {
                continue;
            }
            let iy = condensed_index(n, i, y);
            d[iy] = update_distance(
                method,
                d[condensed_index(n, i, x)],
                d[iy],
                current_min,
                nx,
                ny,
                ni,
            );
        }
    }

    // stable, as SciPy's mergesort
    steps.sort_by(|a, b| a.distance.partial_cmp(&b.distance).expect("finite"));
    relabel(&mut steps, n);
    Ok(steps)
}

///////////
// Tests //
///////////

// the SciPy references are pasted at full precision
#[cfg(test)]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Numerical Recipes LCG points on a small integer grid, so ties occur.
    fn lcg_points(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..2 * n)
            .map(|_| {
                x = (1_664_525 * x + 1_013_904_223) % (1 << 32);
                ((x >> 16) % 7) as f64
            })
            .collect()
    }

    /// Steps as `(left, right, distance, size)` for comparison.
    fn rows(steps: &[LinkageStep<f64>]) -> Vec<(usize, usize, f64, usize)> {
        steps
            .iter()
            .map(|s| (s.left, s.right, s.distance, s.size))
            .collect()
    }

    /// Compare against SciPy 1.18.1 output, generated with
    /// `linkage(pdist(points), method)` on the same LCG points.
    fn assert_matches(got: &[LinkageStep<f64>], want: &[(usize, usize, f64, usize)]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in rows(got).iter().zip(want) {
            assert_eq!((g.0, g.1, g.3), (w.0, w.1, w.3), "{g:?} vs {w:?}");
            assert_relative_eq!(g.2, w.2, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_condensed_euclidean_layout() {
        let data = [0.0, 0.0, 3.0, 4.0, 0.0, 1.0];
        let d = condensed_euclidean(&data, 2);
        assert_eq!(d, vec![5.0, 1.0, 18.0_f64.sqrt()]);
    }

    #[test]
    fn test_condensed_index_matches_pair_order() {
        let n = 5;
        let mut k = 0;
        for i in 0..n {
            for j in i + 1..n {
                assert_eq!(condensed_index(n, i, j), k);
                assert_eq!(condensed_index(n, j, i), k);
                k += 1;
            }
        }
    }

    #[test]
    fn test_linkage_two_points() {
        let steps = linkage(&[2.5_f64], 2, LinkageMethod::Ward).unwrap();
        assert_eq!(rows(&steps), vec![(0, 1, 2.5, 2)]);
    }

    #[test]
    fn test_linkage_rejects_bad_input() {
        assert!(linkage(&[] as &[f64], 1, LinkageMethod::Ward).is_err());
        assert!(linkage(&[1.0_f64, 2.0], 3, LinkageMethod::Ward).is_err());
        assert!(matches!(
            linkage(&[1.0_f64, f64::NAN, 2.0], 3, LinkageMethod::Ward),
            Err(CoreErrors::LinkageNonFiniteDistance { index: 1 })
        ));
    }

    /// Runs every method on 12 LCG points with many tied distances.
    fn run(method: LinkageMethod) -> Vec<LinkageStep<f64>> {
        let points = lcg_points(12, 9);
        linkage(&condensed_euclidean(&points, 2), 12, method).unwrap()
    }

    #[test]
    fn test_linkage_ward_matches_scipy_with_ties() {
        assert_matches(
            &run(LinkageMethod::Ward),
            &[
                (7, 10, 0.0, 2),
                (0, 1, 1.0, 2),
                (3, 11, 1.0, 2),
                (4, 5, 1.0, 2),
                (6, 8, 1.0, 2),
                (12, 14, 1.58113883008418954e+00, 4),
                (9, 15, 2.08166599946613307e+00, 3),
                (2, 13, 3.51188458428424655e+00, 3),
                (17, 18, 3.89444048184930747e+00, 7),
                (16, 20, 5.36449231314369435e+00, 9),
                (19, 21, 8.13770374382246864e+00, 12),
            ],
        );
    }

    #[test]
    fn test_linkage_complete_matches_scipy_with_ties() {
        assert_matches(
            &run(LinkageMethod::Complete),
            &[
                (7, 10, 0.0, 2),
                (0, 1, 1.0, 2),
                (3, 12, 1.0, 3),
                (4, 11, 1.0, 2),
                (6, 8, 1.0, 2),
                (5, 9, 1.41421356237309515e+00, 2),
                (14, 15, 2.23606797749978981e+00, 5),
                (2, 13, 3.16227766016837952e+00, 3),
                (17, 18, 3.16227766016837952e+00, 7),
                (16, 20, 5.09901951359278449e+00, 9),
                (19, 21, 7.07106781186547551e+00, 12),
            ],
        );
    }

    #[test]
    fn test_linkage_average_matches_scipy_with_ties() {
        assert_matches(
            &run(LinkageMethod::Average),
            &[
                (7, 10, 0.0, 2),
                (0, 1, 1.0, 2),
                (3, 12, 1.0, 3),
                (4, 11, 1.0, 2),
                (6, 8, 1.0, 2),
                (5, 15, 1.20710678118654746e+00, 3),
                (9, 17, 1.88342717995762832e+00, 4),
                (14, 18, 2.24435451578731637e+00, 7),
                (2, 13, 3.08113883008418998e+00, 3),
                (16, 19, 3.20798133153849685e+00, 9),
                (20, 21, 4.31326783578189854e+00, 12),
            ],
        );
    }

    #[test]
    fn test_linkage_weighted_matches_scipy_with_ties() {
        assert_matches(
            &run(LinkageMethod::Weighted),
            &[
                (7, 10, 0.0, 2),
                (0, 1, 1.0, 2),
                (3, 12, 1.0, 3),
                (4, 11, 1.0, 2),
                (6, 8, 1.0, 2),
                (5, 15, 1.20710678118654746e+00, 3),
                (9, 17, 1.76612377556149491e+00, 4),
                (14, 16, 2.32514076993644236e+00, 5),
                (2, 13, 3.08113883008418998e+00, 3),
                (18, 19, 3.43154501791733413e+00, 9),
                (20, 21, 4.62563530011323110e+00, 12),
            ],
        );
    }
}
