//! UMAP kNN graph against umap-learn's `fuzzy_simplicial_set`.

mod umap_graph_fixtures;

use bixverse_methods::graph::knn_graph::*;
use umap_graph_fixtures as fx;

/// Numerical Recipes LCG, mirrored from the generator.
fn lcg(n: usize, seed: u32) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            x
        })
        .collect()
}

/// Rebuild the fixture distances from the integer points.
fn distances() -> Vec<f32> {
    let pts: Vec<i64> = lcg(fx::N * fx::DIM, fx::LCG_SEED)
        .into_iter()
        .map(|x| (x % fx::COORD_RANGE) as i64)
        .collect();
    (0..fx::N)
        .flat_map(|i| {
            let pts = &pts;
            fx::IDX[i * fx::K..(i + 1) * fx::K].iter().map(move |&j| {
                let sq: i64 = (0..fx::DIM)
                    .map(|d| (pts[i * fx::DIM + d] - pts[j as usize * fx::DIM + d]).pow(2))
                    .sum();
                (sq as f32).sqrt()
            })
        })
        .collect()
}

/// Same sparsity pattern as umap-learn and weights within f32 rounding.
fn check(local_connectivity: f32, mix_ratio: f32, indptr: &[usize], indices: &[u32], data: &[f32]) {
    let dist = distances();
    let sym = KnnSymmetrisation::Umap {
        local_connectivity: local_connectivity as f64,
        mix_ratio: mix_ratio as f64,
    };
    let g = build_knn_graph(&fx::IDX, Some(&dist), fx::K, fx::N, &sym, 0).unwrap();
    for i in 0..fx::N {
        let (nb, w) = g.get_neighbours_raw(i);
        assert_eq!(nb, &indices[indptr[i]..indptr[i + 1]], "row {i}");
        for (a, b) in w.iter().zip(&data[indptr[i]..indptr[i + 1]]) {
            assert!((a - b).abs() < 1e-5, "row {i}: {a} vs {b}");
        }
    }
}

/// Standard UMAP: fuzzy union at local connectivity 1.
#[test]
fn test_umap_graph_matches_umap_learn_union() {
    check(
        fx::UNION_LOCAL_CONNECTIVITY,
        fx::UNION_MIX_RATIO,
        &fx::UNION_INDPTR,
        &fx::UNION_INDICES,
        &fx::UNION_DATA,
    );
}

/// Fractional local connectivity and a half union, half intersection mix.
#[test]
fn test_umap_graph_matches_umap_learn_mixed() {
    check(
        fx::MIXED_LOCAL_CONNECTIVITY,
        fx::MIXED_MIX_RATIO,
        &fx::MIXED_INDPTR,
        &fx::MIXED_INDICES,
        &fx::MIXED_DATA,
    );
}
