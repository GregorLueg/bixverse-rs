//! Contract checks for values arriving from the R boundary.
//!
//! `bixverse-rs` carries no `#[extendr]` attributes: the R entry points live in
//! the `bixverse` package and reach this crate through a path dependency. That
//! makes the public spatial API the contract, and a value that breaks it shows
//! up as a plausible number in R rather than a compile error.
//!
//! Every check here exists because the offending value is otherwise absorbed
//! rather than rejected. A `NaN` coordinate saturates to the image origin and
//! buckets at `(0, 0)` in the radius graph. A duplicate in `spots_to_keep`
//! shortens the `IndexSet` that defines the local index space while `n_spots`
//! stays put, so expression pairs with the wrong spot's coordinates from the
//! first duplicate onwards. A 1-based neighbour list out of R indexes one past
//! the end.
//!
//! ### Index space
//!
//! `spots_to_keep` is 0-based in the global on-disk space. Neighbour indices,
//! weights and coordinates are 0-based and local, positional against
//! `spots_to_keep`. Nothing here changes that, it only enforces it.

use rustc_hash::FxHashSet;

use crate::prelude::*;

/// Reject coordinates that are not finite.
///
/// Call at any entry point that takes full-res `(x, y)` pairs from R. A
/// `NA_real_` survives the conversion as `NaN` and every consumer downstream
/// swallows it: `NaN <= radius` is false so the spot isolates, `NaN as i32` is
/// 0 so it buckets at the origin, and `(NaN - side / 2.0).round() as i64`
/// saturates to 0 so the tile comes from the top-left corner of the slide.
///
/// ### Params
///
/// * `coordinates` - Per-spot `(x, y)` in full-res pixels.
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::SpatialNonFiniteCoord`] naming the first
/// offending position.
pub fn validate_coordinates(coordinates: &[(f32, f32)]) -> Result<(), BixverseErrors> {
    for (index, &(x, y)) in coordinates.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(BixverseErrors::SpatialNonFiniteCoord { index });
        }
    }
    Ok(())
}

/// Reject a `spots_to_keep` list that repeats a global spot index.
///
/// `n_spots` is `spots_to_keep.len()`, but the local index space is defined by
/// position in the `IndexSet` the readers filter against, and an `IndexSet`
/// deduplicates. A repeat makes the two disagree without ever going out of
/// bounds, so the run completes and returns finite p-values against
/// mis-paired expression.
///
/// ### Params
///
/// * `spots_to_keep` - Global (on-disk) 0-based spot indices for this sample.
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::SpatialDuplicateSpot`] naming the first
/// repeat.
pub fn validate_spots_to_keep(spots_to_keep: &[usize]) -> Result<(), BixverseErrors> {
    let mut seen: FxHashSet<usize> = FxHashSet::default();
    seen.reserve(spots_to_keep.len());
    for (index, &spot) in spots_to_keep.iter().enumerate() {
        if !seen.insert(spot) {
            return Err(BixverseErrors::SpatialDuplicateSpot { spot, index });
        }
    }
    Ok(())
}

/// Reject a neighbour/weight adjacency that does not describe `n_nodes` nodes.
///
/// Checks the three things the CSR build indexes on and never verifies: the
/// two outer lengths agree, every inner pair has the same length, and every
/// neighbour index is local. A 1-based neighbour list out of R fails the last
/// one, which otherwise panics with an opaque out-of-bounds inside
/// `make_weights_non_redundant`.
///
/// ### Params
///
/// * `neighbours` - Per-node local neighbour indices, `0..n_nodes`.
/// * `weights` - Per-node edge weights, paired positionally with `neighbours`.
/// * `n_nodes` - Expected node count.
///
/// ### Returns
///
/// `Ok(())`, or the first violation as
/// [`BixverseErrors::SpatialGraphSpotMismatch`],
/// [`BixverseErrors::SpatialAdjacencyRagged`] or
/// [`BixverseErrors::SpatialNeighbourOutOfRange`].
pub fn validate_adjacency(
    neighbours: &[Vec<usize>],
    weights: &[Vec<f32>],
    n_nodes: usize,
) -> Result<(), BixverseErrors> {
    if neighbours.len() != n_nodes {
        return Err(BixverseErrors::SpatialGraphSpotMismatch {
            graph_nodes: neighbours.len(),
            n_spots: n_nodes,
        });
    }
    if weights.len() != n_nodes {
        return Err(BixverseErrors::SpatialGraphSpotMismatch {
            graph_nodes: weights.len(),
            n_spots: n_nodes,
        });
    }

    for (node, (neigh, w)) in neighbours.iter().zip(weights.iter()).enumerate() {
        if neigh.len() != w.len() {
            return Err(BixverseErrors::SpatialAdjacencyRagged {
                node,
                n_neighbours: neigh.len(),
                n_weights: w.len(),
            });
        }
        for &j in neigh {
            if j >= n_nodes {
                return Err(BixverseErrors::SpatialNeighbourOutOfRange {
                    node,
                    neighbour: j,
                    n_nodes,
                });
            }
        }
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_coordinates_accepts_finite() {
        assert!(validate_coordinates(&[(0.0, 0.0), (1.5, -2.5)]).is_ok());
    }

    #[test]
    fn test_validate_coordinates_rejects_nan() {
        let err = validate_coordinates(&[(1.0, 1.0), (f32::NAN, 2.0)]).unwrap_err();
        assert!(matches!(
            err,
            BixverseErrors::SpatialNonFiniteCoord { index: 1 }
        ));
    }

    #[test]
    fn test_validate_coordinates_rejects_infinity() {
        let err = validate_coordinates(&[(1.0, f32::INFINITY)]).unwrap_err();
        assert!(matches!(
            err,
            BixverseErrors::SpatialNonFiniteCoord { index: 0 }
        ));
    }

    #[test]
    fn test_validate_spots_to_keep_accepts_unique() {
        assert!(validate_spots_to_keep(&[5, 7, 9]).is_ok());
    }

    #[test]
    fn test_validate_spots_to_keep_rejects_duplicate() {
        // The review's case: n_spots = 4 but the IndexSet holds {5, 7, 9}.
        let err = validate_spots_to_keep(&[5, 7, 5, 9]).unwrap_err();
        match err {
            BixverseErrors::SpatialDuplicateSpot { spot, index } => {
                assert_eq!(spot, 5);
                assert_eq!(index, 2);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn test_validate_adjacency_accepts_local_indices() {
        let neigh = vec![vec![1], vec![0, 2], vec![1]];
        let w = vec![vec![1.0_f32], vec![1.0, 1.0], vec![1.0]];
        assert!(validate_adjacency(&neigh, &w, 3).is_ok());
    }

    #[test]
    fn test_validate_adjacency_rejects_one_based_indices() {
        // What a neighbour list straight out of R looks like.
        let neigh = vec![vec![2], vec![1, 3], vec![2]];
        let w = vec![vec![1.0_f32], vec![1.0, 1.0], vec![1.0]];
        let err = validate_adjacency(&neigh, &w, 3).unwrap_err();
        match err {
            BixverseErrors::SpatialNeighbourOutOfRange {
                node,
                neighbour,
                n_nodes,
            } => {
                assert_eq!(node, 1);
                assert_eq!(neighbour, 3);
                assert_eq!(n_nodes, 3);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn test_validate_adjacency_rejects_ragged_rows() {
        let neigh = vec![vec![1], vec![0, 2], vec![1]];
        let w = vec![vec![1.0_f32], vec![1.0], vec![1.0]];
        let err = validate_adjacency(&neigh, &w, 3).unwrap_err();
        assert!(matches!(
            err,
            BixverseErrors::SpatialAdjacencyRagged { node: 1, .. }
        ));
    }

    #[test]
    fn test_validate_adjacency_rejects_node_count_mismatch() {
        let neigh = vec![vec![1], vec![0]];
        let w = vec![vec![1.0_f32], vec![1.0]];
        assert!(matches!(
            validate_adjacency(&neigh, &w, 3).unwrap_err(),
            BixverseErrors::SpatialGraphSpotMismatch { .. }
        ));
    }
}
