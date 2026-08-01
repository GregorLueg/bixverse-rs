//! Spatial adjacency construction from spot coordinates.
//!
//! Four layouts. The two Visium lattices are exact: the capture geometry is
//! known, so a neighbour is a table lookup and no distance ever gets computed
//! to find it. kNN and fixed-radius cover irregular geometries (Xenium,
//! MERFISH) where there is no lattice to exploit.
//!
//! Everything is `f32` rather than generic over the float, because the two
//! consumers downstream, [`GraphCsr`] and `SparkX`, already are.
//!
//! ### Index space
//!
//! Coordinates come in as full-res pixels, `(x, y)`. The returned neighbour
//! and weight lists are local and 0-based: entry `i` describes the spot at
//! `coordinates[i]`, and the indices inside it position against that same
//! slice. `MoransI::new` checks `neighbours.len() == spots_to_keep.len()`,
//! which is what makes them local rather than global spot ids.
//!
//! ### Asymmetry
//!
//! kNN adjacency is directed and row-normalisation makes any layout directed.
//! Neither is a problem: [`make_weights_non_redundant`] collapses `(i, j)` and
//! `(j, i)` into `w_ij + w_ji` before [`GraphCsr::from_non_redundant`]
//! scatters it back into both rows, so the graph the statistics see is always
//! `W + Wᵀ`.

use faer::Mat;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::math::vector_helpers::median;
use crate::graph::spatial_graph::{GraphCsr, make_weights_non_redundant};
use crate::prelude::*;

///////////
// Types //
///////////

/// The `(neighbours, weights)` adjacency pair the spatial statistics consume.
///
/// Entry `i` of each vector describes spot `i`, and the two are aligned
/// element for element: `weights[i][k]` is the weight of the edge to
/// `neighbours[i][k]`.
pub type SpatialAdjacency = (Vec<Vec<usize>>, Vec<Vec<f32>>);

///////////////
// Constants //
///////////////

/// Neighbour offsets in `(array_row, array_col)` on the Visium honeycomb.
///
/// Odd rows are staggered half a column against even rows, and 10x encodes
/// that by stepping `array_col` in twos. So the two in-row neighbours sit at
/// `col ± 2` and the four out-of-row ones at `(row ± 1, col ± 1)`. A corollary
/// is that `array_row % 2 == array_col % 2` for every spot, which is why the
/// offsets never leave the lattice.
const HEX_OFFSETS: [(i32, i32); 6] = [(-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1)];

/// Von Neumann (4-connected) offsets on a square bin lattice.
const SQUARE_OFFSETS_4: [(i32, i32); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

/// Moore (8-connected) offsets on a square bin lattice.
const SQUARE_OFFSETS_8: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Distance floor for inverse-distance weighting.
///
/// Two spots at identical coordinates would otherwise give an infinite weight,
/// which poisons the row sum and with it the row-normalisation. Clamping keeps
/// the row finite. Coincident spots are a segmentation artefact rather than a
/// legitimate input, so the exact value does not matter much.
const MIN_EDGE_DISTANCE: f32 = 1e-6;

//////////////
// Settings //
//////////////

/// How adjacency is derived.
///
/// Lattice variants are exact and need no distance computation; the rest fall
/// back to geometry.
#[derive(Clone, Debug)]
pub enum SpatialGraphLayout {
    /// Visium honeycomb. Neighbours of `(r, c)` are `(r, c ± 2)` and
    /// `(r ± 1, c ± 1)`.
    HexLattice {
        /// Per-spot `array_row` from the 10x tissue positions table.
        array_row: Vec<i32>,
        /// Per-spot `array_col` from the 10x tissue positions table.
        array_col: Vec<i32>,
    },
    /// Visium HD square bins. 4- or 8-connected.
    SquareLattice {
        /// Per-spot bin row index.
        array_row: Vec<i32>,
        /// Per-spot bin column index.
        array_col: Vec<i32>,
        /// Either `4` (von Neumann) or `8` (Moore).
        connectivity: u8,
    },
    /// Irregular geometry (Xenium, MERFISH). Delegates to `ann-search-rs`.
    Knn {
        /// Number of neighbours per spot, self excluded.
        k: usize,
    },
    /// Fixed radius in full-res pixels, uniform grid buckets.
    Radius {
        /// Cut-off distance. Edges are kept when `d <= radius`.
        radius: f32,
    },
}

/// How an edge distance becomes an edge weight.
#[derive(Clone, Copy, Debug)]
pub enum SpatialWeighting {
    /// Every edge weighs 1. The only sensible choice for a lattice, where all
    /// edges are the same length anyway.
    Binary,
    /// `w = d^-power`, with `d` clamped at [`MIN_EDGE_DISTANCE`].
    InverseDistance {
        /// Decay exponent. `1.0` and `2.0` are the usual picks.
        power: f32,
    },
    /// `w = exp(-d^2 / (2 * bandwidth^2))`, matching the SPARK-X Gaussian
    /// kernel convention in this crate.
    Gaussian {
        /// `None` bandwidth resolves to the median nearest-neighbour distance.
        bandwidth: Option<f32>,
    },
}

/// Parameters for [`build_spatial_graph`].
#[derive(Clone, Debug)]
pub struct SpatialGraphParams {
    /// Which adjacency rule to apply.
    pub layout: SpatialGraphLayout,
    /// Which weighting to apply on top of it.
    pub weighting: SpatialWeighting,
    /// spdep `style = "W"`. Row-normalise so each row sums to 1.
    pub row_normalise: bool,
}

impl SpatialGraphParams {
    /// Build a parameter set.
    ///
    /// ### Params
    ///
    /// * `layout` - Adjacency rule.
    /// * `weighting` - Edge weighting.
    /// * `row_normalise` - Row-normalise the weights.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new(
        layout: SpatialGraphLayout,
        weighting: SpatialWeighting,
        row_normalise: bool,
    ) -> Self {
        Self {
            layout,
            weighting,
            row_normalise,
        }
    }
}

/// Default implementation. Visium-shaped: 6-nearest-neighbour, binary, no
/// row-normalisation.
impl Default for SpatialGraphParams {
    fn default() -> Self {
        Self {
            layout: SpatialGraphLayout::Knn { k: 6 },
            weighting: SpatialWeighting::Binary,
            row_normalise: false,
        }
    }
}

///////////////////
// Entry  points //
///////////////////

/// Build the spatial neighbourhood graph for one sample.
///
/// Returns exactly the pair `MoransI::new` takes. Neighbour lists come back
/// sorted ascending, so the output is canonical and does not depend on hash
/// iteration order or on which ANN backend ran.
///
/// Fewer than two coordinates gives an empty adjacency rather than an error:
/// a single-spot sample has no edges to build and the caller's shape checks
/// still line up.
///
/// ### Params
///
/// * `coordinates` - Per-spot full-res pixel coordinates, `(x, y)`.
/// * `params` - Layout, weighting and normalisation choice.
/// * `knn_params` - ANN settings for [`SpatialGraphLayout::Knn`]. Ignored by
///   every other layout. `None` falls back to [`KnnParams::default`]. The `k`
///   and `ann_dist` fields are overridden from the layout and by the 2D
///   geometry respectively.
/// * `seed` - Seed for the ANN index, for reproducibility.
///
/// ### Returns
///
/// `(neighbours, weights)`, both local and 0-based against `coordinates`.
pub fn build_spatial_graph(
    coordinates: &[(f32, f32)],
    params: &SpatialGraphParams,
    knn_params: Option<&KnnParams>,
    seed: usize,
) -> Result<SpatialAdjacency, BixverseErrors> {
    let n = coordinates.len();

    let neighbours = match &params.layout {
        SpatialGraphLayout::HexLattice {
            array_row,
            array_col,
        } => lattice_neighbours(array_row, array_col, n, &HEX_OFFSETS)?,
        SpatialGraphLayout::SquareLattice {
            array_row,
            array_col,
            connectivity,
        } => match connectivity {
            4 => lattice_neighbours(array_row, array_col, n, &SQUARE_OFFSETS_4)?,
            8 => lattice_neighbours(array_row, array_col, n, &SQUARE_OFFSETS_8)?,
            other => {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "square lattice connectivity must be 4 or 8, got {other}"
                )));
            }
        },
        SpatialGraphLayout::Knn { k } => knn_neighbours(coordinates, *k, knn_params, seed)?,
        SpatialGraphLayout::Radius { radius } => radius_neighbours(coordinates, *radius)?,
    };

    let distances = edge_distances(coordinates, &neighbours);
    let mut weights = apply_weighting(&distances, &params.weighting)?;

    if params.row_normalise {
        row_normalise_inplace(&mut weights);
    }

    Ok((neighbours, weights))
}

/// [`build_spatial_graph`] straight into the symmetric CSR.
///
/// Saves the caller repeating the [`make_weights_non_redundant`] then
/// [`GraphCsr::from_non_redundant`] dance that neighbourhood enrichment and
/// Moran's I both need.
///
/// ### Params
///
/// * `coordinates` - Per-spot full-res pixel coordinates, `(x, y)`.
/// * `params` - Layout, weighting and normalisation choice.
/// * `knn_params` - ANN settings for [`SpatialGraphLayout::Knn`].
/// * `seed` - Seed for the ANN index.
///
/// ### Returns
///
/// The symmetric [`GraphCsr`] over the same local index space.
pub fn build_spatial_graph_csr(
    coordinates: &[(f32, f32)],
    params: &SpatialGraphParams,
    knn_params: Option<&KnnParams>,
    seed: usize,
) -> Result<GraphCsr, BixverseErrors> {
    let (neighbours, weights) = build_spatial_graph(coordinates, params, knn_params, seed)?;
    let non_redundant = make_weights_non_redundant(&neighbours, &weights);
    Ok(GraphCsr::from_non_redundant(&neighbours, &non_redundant))
}

///////////////
// Adjacency //
///////////////

/// Neighbours on an integer lattice by probing a fixed offset set.
///
/// Exact and linear in `n * offsets.len()`. Builds a lookup from lattice
/// coordinate to spot index once, then probes. No distances involved, which is
/// the whole point of having a lattice.
///
/// ### Params
///
/// * `array_row` - Per-spot lattice row.
/// * `array_col` - Per-spot lattice column.
/// * `n_spots` - Number of coordinates, for the length check.
/// * `offsets` - `(row, col)` deltas defining the neighbourhood.
///
/// ### Returns
///
/// Sorted neighbour lists. Spots on an edge or corner simply get fewer
/// neighbours; there is no padding or wrap-around.
fn lattice_neighbours(
    array_row: &[i32],
    array_col: &[i32],
    n_spots: usize,
    offsets: &[(i32, i32)],
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    if array_row.len() != n_spots || array_col.len() != n_spots {
        return Err(BixverseErrors::SpatialLatticeLengthMismatch {
            n_array_row: array_row.len(),
            n_array_col: array_col.len(),
            n_spots,
        });
    }

    let mut lookup: FxHashMap<(i32, i32), usize> = FxHashMap::default();
    lookup.reserve(n_spots);
    for i in 0..n_spots {
        let key = (array_row[i], array_col[i]);
        if lookup.insert(key, i).is_some() {
            return Err(BixverseErrors::SpatialLatticeDuplicateCoord {
                array_row: key.0,
                array_col: key.1,
            });
        }
    }

    let neighbours = (0..n_spots)
        .into_par_iter()
        .map(|i| {
            let (r, c) = (array_row[i], array_col[i]);
            let mut out: Vec<usize> = offsets
                .iter()
                .filter_map(|&(dr, dc)| lookup.get(&(r + dr, c + dc)).copied())
                .collect();
            out.sort_unstable();
            out
        })
        .collect();

    Ok(neighbours)
}

/// Neighbours within a fixed radius, via uniform grid buckets.
///
/// Cell size equals the radius, so every candidate within range lives in the
/// 3x3 block around the query cell. Buckets are a hash map keyed by cell
/// rather than a dense grid, which keeps memory bounded by `n` no matter how
/// small the radius or how skewed the bounding box.
///
/// ### Params
///
/// * `coordinates` - Per-spot `(x, y)`.
/// * `radius` - Cut-off distance in the same units as the coordinates.
///
/// ### Returns
///
/// Sorted neighbour lists. Symmetric by construction, since `d(i, j) <= r` is.
fn radius_neighbours(
    coordinates: &[(f32, f32)],
    radius: f32,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(BixverseErrors::InvalidArgument(format!(
            "radius must be finite and greater than zero, got {radius}"
        )));
    }

    let n = coordinates.len();
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }

    let (min_x, min_y) = coordinates
        .iter()
        .fold((f32::INFINITY, f32::INFINITY), |(ax, ay), &(x, y)| {
            (ax.min(x), ay.min(y))
        });

    let cell_of = |(x, y): (f32, f32)| -> (i32, i32) {
        (
            ((x - min_x) / radius).floor() as i32,
            ((y - min_y) / radius).floor() as i32,
        )
    };

    let mut buckets: FxHashMap<(i32, i32), Vec<usize>> = FxHashMap::default();
    for (i, &c) in coordinates.iter().enumerate() {
        buckets.entry(cell_of(c)).or_default().push(i);
    }

    let radius_sq = radius * radius;

    let neighbours = (0..n)
        .into_par_iter()
        .map(|i| {
            let (xi, yi) = coordinates[i];
            let (cx, cy) = cell_of(coordinates[i]);
            let mut out: Vec<usize> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(bucket) = buckets.get(&(cx + dx, cy + dy)) else {
                        continue;
                    };
                    for &j in bucket {
                        if j == i {
                            continue;
                        }
                        let (xj, yj) = coordinates[j];
                        let d_sq = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
                        if d_sq <= radius_sq {
                            out.push(j);
                        }
                    }
                }
            }
            out.sort_unstable();
            out
        })
        .collect();

    Ok(neighbours)
}

/// Neighbours from an approximate nearest-neighbour search over the 2D
/// coordinates.
///
/// Hands an `n x 2` matrix to [`generate_knn_with_dist`] rather than rolling a
/// second ANN path. `ann_dist` is forced to `"euclidean"` because the default
/// on [`KnnParams`] is cosine, which is meaningless on positions, and `k` is
/// taken from the layout and clamped to `n - 1`.
///
/// Distances that come back are discarded. They are squared Euclidean in
/// `ann-search-rs`, and recomputing the true distance from the coordinates
/// costs one `sqrt` per edge, so the weighting path stays identical across all
/// four layouts instead of special-casing this one.
///
/// ### Params
///
/// * `coordinates` - Per-spot `(x, y)`.
/// * `k` - Neighbours per spot, self excluded.
/// * `knn_params` - ANN settings, `None` for [`KnnParams::default`].
/// * `seed` - Seed for the index.
///
/// ### Returns
///
/// Sorted neighbour lists. Directed: `j` in row `i` does not imply the
/// reverse.
fn knn_neighbours(
    coordinates: &[(f32, f32)],
    k: usize,
    knn_params: Option<&KnnParams>,
    seed: usize,
) -> Result<Vec<Vec<usize>>, BixverseErrors> {
    if k == 0 {
        return Err(BixverseErrors::MustBePositive("k".to_string()));
    }

    let n = coordinates.len();
    if n < 2 {
        return Ok(vec![Vec::new(); n]);
    }

    let mut params = knn_params.cloned().unwrap_or_default();
    params.k = k.min(n - 1);
    params.ann_dist = "euclidean".to_string();

    let embedding = Mat::<f32>::from_fn(n, 2, |i, j| {
        if j == 0 {
            coordinates[i].0
        } else {
            coordinates[i].1
        }
    });

    let (mut indices, _) =
        generate_knn_with_dist(embedding.as_ref(), &params, false, false, seed, false)?;

    for row in indices.iter_mut() {
        row.sort_unstable();
    }

    Ok(indices)
}

///////////////
// Weighting //
///////////////

/// Euclidean distance for every edge in the adjacency.
///
/// ### Params
///
/// * `coordinates` - Per-spot `(x, y)`.
/// * `neighbours` - Adjacency lists.
///
/// ### Returns
///
/// Distances aligned element-for-element with `neighbours`.
fn edge_distances(coordinates: &[(f32, f32)], neighbours: &[Vec<usize>]) -> Vec<Vec<f32>> {
    neighbours
        .par_iter()
        .enumerate()
        .map(|(i, nb)| {
            let (xi, yi) = coordinates[i];
            nb.iter()
                .map(|&j| {
                    let (xj, yj) = coordinates[j];
                    ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)).sqrt()
                })
                .collect()
        })
        .collect()
}

/// Median nearest-neighbour distance across spots.
///
/// The heuristic behind `Gaussian { bandwidth: None }`. Takes the shortest
/// edge per spot, then the median over spots, so a handful of isolated spots
/// with abnormally long edges cannot drag the bandwidth up. Spots with no
/// edges contribute nothing.
///
/// ### Params
///
/// * `distances` - Per-spot edge distances.
///
/// ### Returns
///
/// The bandwidth, or an error when the graph carries no usable edge.
fn resolve_bandwidth(distances: &[Vec<f32>]) -> Result<f32, BixverseErrors> {
    let nn_distances: Vec<f32> = distances
        .iter()
        .filter_map(|d| {
            d.iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(None, |acc: Option<f32>, v| {
                    Some(acc.map_or(v, |a| a.min(v)))
                })
        })
        .collect();

    match median(&nn_distances) {
        Some(m) if m > 0.0 => Ok(m),
        _ => Err(BixverseErrors::SpatialGraphBandwidthUnresolved),
    }
}

/// Map edge distances onto edge weights.
///
/// ### Params
///
/// * `distances` - Per-spot edge distances.
/// * `weighting` - Which kernel to apply.
///
/// ### Returns
///
/// Weights aligned element-for-element with `distances`.
fn apply_weighting(
    distances: &[Vec<f32>],
    weighting: &SpatialWeighting,
) -> Result<Vec<Vec<f32>>, BixverseErrors> {
    let weights = match weighting {
        SpatialWeighting::Binary => distances.iter().map(|d| vec![1.0_f32; d.len()]).collect(),
        SpatialWeighting::InverseDistance { power } => distances
            .par_iter()
            .map(|d| {
                d.iter()
                    .map(|&v| v.max(MIN_EDGE_DISTANCE).powf(-power))
                    .collect()
            })
            .collect(),
        SpatialWeighting::Gaussian { bandwidth } => {
            let bandwidth = match bandwidth {
                Some(b) if *b > 0.0 && b.is_finite() => *b,
                Some(b) => {
                    return Err(BixverseErrors::InvalidArgument(format!(
                        "Gaussian bandwidth must be finite and greater than zero, got {b}"
                    )));
                }
                None => resolve_bandwidth(distances)?,
            };
            let denominator = 2.0 * bandwidth * bandwidth;
            distances
                .par_iter()
                .map(|d| d.iter().map(|&v| (-(v * v) / denominator).exp()).collect())
                .collect()
        }
    };

    Ok(weights)
}

/// Divide each row by its sum, giving spdep's `style = "W"`.
///
/// Rows that sum to zero (an isolated spot, or a Gaussian kernel that
/// underflowed) are left alone. Scaling them is undefined and zeroed rows drop
/// out of [`GraphCsr::from_non_redundant`] anyway.
///
/// ### Params
///
/// * `weights` - Weights, modified in place.
fn row_normalise_inplace(weights: &mut [Vec<f32>]) {
    weights.par_iter_mut().for_each(|row| {
        let total: f32 = row.iter().sum();
        if total > 0.0 && total.is_finite() {
            for w in row.iter_mut() {
                *w /= total;
            }
        }
    });
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::sp_processing::morans_i::MoransI;
    use crate::spatial::sp_processing::svg_utils::{EmptyGeneReader, SpatialSvgParams};

    /// Visium spacing: a hexagon of unit side, centre plus its six neighbours.
    /// Lattice coordinates follow the 10x convention, so `array_col` steps in
    /// twos along a row and `array_row % 2 == array_col % 2` throughout.
    fn hexagon() -> (Vec<(f32, f32)>, Vec<i32>, Vec<i32>) {
        // (row, col) -> (x, y) with x = col * 0.5, y = row * (sqrt(3) / 2).
        let lattice = [
            (2_i32, 2_i32), // centre, index 0
            (2, 0),
            (2, 4),
            (1, 1),
            (1, 3),
            (3, 1),
            (3, 3),
        ];
        let y_step = 3.0_f32.sqrt() / 2.0;
        let coords: Vec<(f32, f32)> = lattice
            .iter()
            .map(|&(r, c)| (c as f32 * 0.5, r as f32 * y_step))
            .collect();
        let rows = lattice.iter().map(|&(r, _)| r).collect();
        let cols = lattice.iter().map(|&(_, c)| c).collect();
        (coords, rows, cols)
    }

    /// `side x side` grid of unit spacing, row-major.
    fn square_grid(side: usize) -> Vec<(f32, f32)> {
        (0..side)
            .flat_map(|r| (0..side).map(move |c| (c as f32, r as f32)))
            .collect()
    }

    #[test]
    fn hex_lattice_centre_spot_has_degree_six() {
        let (coords, array_row, array_col) = hexagon();
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row,
                array_col,
            },
            SpatialWeighting::Binary,
            false,
        );

        let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        assert_eq!(neighbours.len(), 7);
        assert_eq!(neighbours[0], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(weights[0], vec![1.0; 6]);

        // The ring closes on itself, so each outer spot sees the centre plus
        // its two ring-mates: degree 3, never 1.
        for i in 1..7 {
            assert_eq!(neighbours[i].len(), 3, "spot {i}");
            assert!(neighbours[i].contains(&0), "spot {i}");
        }

        // Every edge the rule produced is a unit hexagon side. That is the
        // check that the offsets encode the real geometry rather than some
        // lattice artefact.
        for (i, nb) in neighbours.iter().enumerate() {
            for &j in nb {
                let (xi, yi) = coords[i];
                let (xj, yj) = coords[j];
                let d = ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)).sqrt();
                approx::assert_relative_eq!(d, 1.0, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn hex_lattice_interior_degree_six_on_a_larger_patch() {
        // 9 rows x 9 lattice columns of the honeycomb, keeping only the spots
        // that satisfy the 10x parity rule. Interior spots must all land on
        // degree 6; nothing in between is allowed.
        let mut coords = Vec::new();
        let mut array_row = Vec::new();
        let mut array_col = Vec::new();
        let y_step = 3.0_f32.sqrt() / 2.0;
        for r in 0..9_i32 {
            for c in 0..18_i32 {
                if r.rem_euclid(2) != c.rem_euclid(2) {
                    continue;
                }
                coords.push((c as f32 * 0.5, r as f32 * y_step));
                array_row.push(r);
                array_col.push(c);
            }
        }

        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row: array_row.clone(),
                array_col: array_col.clone(),
            },
            SpatialWeighting::Binary,
            false,
        );
        let (neighbours, _) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        let mut interior = 0_usize;
        for i in 0..coords.len() {
            let (r, c) = (array_row[i], array_col[i]);
            let on_border = r == 0 || r == 8 || c <= 1 || c >= 16;
            if on_border {
                assert!(neighbours[i].len() < 6, "border spot {i}");
            } else {
                assert_eq!(neighbours[i].len(), 6, "interior spot {i}");
                interior += 1;
            }
        }
        assert!(interior > 0);
    }

    #[test]
    fn hex_lattice_ignores_wrong_parity_spots() {
        // Append a spot at (2, 3): parity broken, so no offset can reach it and
        // it can reach nothing.
        let (mut coords, mut array_row, mut array_col) = hexagon();
        coords.push((1.5, 3.0_f32.sqrt()));
        array_row.push(2);
        array_col.push(3);

        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row,
                array_col,
            },
            SpatialWeighting::Binary,
            false,
        );

        let (neighbours, _) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        assert!(neighbours[7].is_empty());
        assert_eq!(neighbours[0].len(), 6);
        assert!(!neighbours[0].contains(&7));
    }

    #[test]
    fn square_lattice_connectivity_controls_degree() {
        let side = 3_usize;
        let coords = square_grid(side);
        let array_row: Vec<i32> = (0..side)
            .flat_map(|r| (0..side).map(move |_| r as i32))
            .collect();
        let array_col: Vec<i32> = (0..side)
            .flat_map(|_| (0..side).map(|c| c as i32))
            .collect();
        let centre = 4_usize;

        for (connectivity, expected) in [(4_u8, 4_usize), (8, 8)] {
            let params = SpatialGraphParams::new(
                SpatialGraphLayout::SquareLattice {
                    array_row: array_row.clone(),
                    array_col: array_col.clone(),
                    connectivity,
                },
                SpatialWeighting::Binary,
                false,
            );
            let (neighbours, _) = build_spatial_graph(&coords, &params, None, 42).unwrap();
            assert_eq!(
                neighbours[centre].len(),
                expected,
                "connectivity {connectivity}"
            );
        }
    }

    #[test]
    fn square_lattice_rejects_other_connectivity() {
        let coords = square_grid(2);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::SquareLattice {
                array_row: vec![0, 0, 1, 1],
                array_col: vec![0, 1, 0, 1],
                connectivity: 6,
            },
            SpatialWeighting::Binary,
            false,
        );
        assert!(matches!(
            build_spatial_graph(&coords, &params, None, 42),
            Err(BixverseErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn lattice_length_mismatch_errors() {
        let coords = square_grid(2);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row: vec![0, 0, 2],
                array_col: vec![0, 2, 0],
            },
            SpatialWeighting::Binary,
            false,
        );
        match build_spatial_graph(&coords, &params, None, 42) {
            Err(BixverseErrors::SpatialLatticeLengthMismatch {
                n_array_row,
                n_array_col,
                n_spots,
            }) => {
                assert_eq!(n_array_row, 3);
                assert_eq!(n_array_col, 3);
                assert_eq!(n_spots, 4);
            }
            other => panic!("wrong outcome: {other:?}"),
        }
    }

    #[test]
    fn lattice_duplicate_coordinate_errors() {
        let coords = square_grid(2);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row: vec![0, 0, 2, 2],
                array_col: vec![0, 0, 0, 2],
            },
            SpatialWeighting::Binary,
            false,
        );
        assert!(matches!(
            build_spatial_graph(&coords, &params, None, 42),
            Err(BixverseErrors::SpatialLatticeDuplicateCoord { .. })
        ));
    }

    #[test]
    fn knn_and_radius_agree_on_uniform_grid() {
        // 5x5 unit grid. For an interior spot the four nearest are the axis
        // neighbours at d = 1; the diagonals sit at d = 1.414. A radius of 1.01
        // picks out exactly the same four, so interior rows must match. Border
        // spots have fewer than four axis neighbours and kNN pads them with
        // diagonals, hence the interior-only comparison.
        let side = 5_usize;
        let coords = square_grid(side);

        let knn_settings = KnnParams {
            knn_method: "exhaustive".to_string(),
            ..KnnParams::default()
        };

        let knn = build_spatial_graph(
            &coords,
            &SpatialGraphParams::new(
                SpatialGraphLayout::Knn { k: 4 },
                SpatialWeighting::Binary,
                false,
            ),
            Some(&knn_settings),
            42,
        )
        .unwrap()
        .0;

        let radius = build_spatial_graph(
            &coords,
            &SpatialGraphParams::new(
                SpatialGraphLayout::Radius { radius: 1.01 },
                SpatialWeighting::Binary,
                false,
            ),
            None,
            42,
        )
        .unwrap()
        .0;

        for r in 1..side - 1 {
            for c in 1..side - 1 {
                let i = r * side + c;
                assert_eq!(radius[i].len(), 4, "spot {i}");
                assert_eq!(knn[i], radius[i], "spot {i}");
            }
        }
    }

    #[test]
    fn radius_isolates_spots_beyond_cut_off() {
        let coords = vec![(0.0_f32, 0.0), (1.0, 0.0), (100.0, 100.0)];
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::Binary,
            false,
        );
        let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        assert_eq!(neighbours[0], vec![1]);
        assert_eq!(neighbours[1], vec![0]);
        assert!(neighbours[2].is_empty());
        assert!(weights[2].is_empty());
    }

    #[test]
    fn radius_rejects_non_positive_cut_off() {
        let coords = square_grid(3);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 0.0 },
            SpatialWeighting::Binary,
            false,
        );
        assert!(matches!(
            build_spatial_graph(&coords, &params, None, 42),
            Err(BixverseErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn row_normalisation_sums_to_one() {
        let coords = square_grid(5);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::InverseDistance { power: 1.0 },
            true,
        );
        let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        for (i, row) in weights.iter().enumerate() {
            assert!(!neighbours[i].is_empty());
            let total: f32 = row.iter().sum();
            approx::assert_relative_eq!(total, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn inverse_distance_weights_decay_with_distance() {
        // Diagonals sit at sqrt(2), axis neighbours at 1, so with power 2 the
        // diagonal weight is exactly half.
        let coords = square_grid(3);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::InverseDistance { power: 2.0 },
            false,
        );
        let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        let centre = 4_usize;
        for (k, &j) in neighbours[centre].iter().enumerate() {
            let expected = if j % 2 == 0 { 0.5 } else { 1.0 };
            approx::assert_relative_eq!(weights[centre][k], expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn gaussian_bandwidth_resolves_to_median_nearest_neighbour() {
        let coords = square_grid(4);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::Gaussian { bandwidth: None },
            false,
        );
        let (neighbours, auto) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        // Every spot on a unit grid has a nearest neighbour at exactly 1, so
        // the median resolves to 1 and the explicit run must match.
        let explicit = build_spatial_graph(
            &coords,
            &SpatialGraphParams::new(
                SpatialGraphLayout::Radius { radius: 1.5 },
                SpatialWeighting::Gaussian {
                    bandwidth: Some(1.0),
                },
                false,
            ),
            None,
            42,
        )
        .unwrap()
        .1;

        for i in 0..neighbours.len() {
            for k in 0..neighbours[i].len() {
                approx::assert_relative_eq!(auto[i][k], explicit[i][k], epsilon = 1e-6);
            }
        }
        // exp(-1 / 2) for an axis neighbour at d = 1.
        let centre = 5_usize;
        let axis = neighbours[centre].iter().position(|&j| j == 4).unwrap();
        approx::assert_relative_eq!(auto[centre][axis], (-0.5_f32).exp(), epsilon = 1e-5);
    }

    #[test]
    fn gaussian_rejects_non_positive_bandwidth() {
        let coords = square_grid(3);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::Gaussian {
                bandwidth: Some(-1.0),
            },
            false,
        );
        assert!(matches!(
            build_spatial_graph(&coords, &params, None, 42),
            Err(BixverseErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn output_feeds_morans_i_without_shape_error() {
        let (coords, array_row, array_col) = hexagon();
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::HexLattice {
                array_row,
                array_col,
            },
            SpatialWeighting::Binary,
            true,
        );
        let (neighbours, mut weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();

        let spots: Vec<usize> = (0..coords.len()).collect();
        let morans = MoransI::new(
            &EmptyGeneReader,
            &spots,
            &neighbours,
            &mut weights,
            SpatialSvgParams::default(),
        )
        .unwrap();

        assert_eq!(morans.n_spots(), coords.len());
        approx::assert_relative_eq!(morans.e_i(), -1.0 / 6.0, epsilon = 1e-9);
        assert!(morans.var_i().is_finite());
    }

    #[test]
    fn csr_convenience_matches_the_manual_path() {
        let coords = square_grid(4);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Radius { radius: 1.5 },
            SpatialWeighting::Binary,
            false,
        );

        let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();
        let non_redundant = make_weights_non_redundant(&neighbours, &weights);
        let expected = GraphCsr::from_non_redundant(&neighbours, &non_redundant);

        let graph = build_spatial_graph_csr(&coords, &params, None, 42).unwrap();

        assert_eq!(graph.n_nodes(), expected.n_nodes());
        assert_eq!(graph.nnz(), expected.nnz());
        assert_eq!(graph.offsets(), expected.offsets());
        assert_eq!(graph.indices(), expected.indices());
        approx::assert_relative_eq!(graph.weight_sum(), expected.weight_sum(), epsilon = 1e-9);
    }

    #[test]
    fn single_spot_gives_an_empty_adjacency() {
        let coords = vec![(0.0_f32, 0.0)];
        for layout in [
            SpatialGraphLayout::Knn { k: 6 },
            SpatialGraphLayout::Radius { radius: 10.0 },
        ] {
            let params = SpatialGraphParams::new(layout, SpatialWeighting::Binary, true);
            let (neighbours, weights) = build_spatial_graph(&coords, &params, None, 42).unwrap();
            assert_eq!(neighbours.len(), 1);
            assert!(neighbours[0].is_empty());
            assert!(weights[0].is_empty());
        }
    }

    #[test]
    fn knn_rejects_zero_k() {
        let coords = square_grid(3);
        let params = SpatialGraphParams::new(
            SpatialGraphLayout::Knn { k: 0 },
            SpatialWeighting::Binary,
            false,
        );
        assert!(matches!(
            build_spatial_graph(&coords, &params, None, 42),
            Err(BixverseErrors::MustBePositive(_))
        ));
    }
}
