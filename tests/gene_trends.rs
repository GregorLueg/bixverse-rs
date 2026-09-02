//! End-to-end trajectory test: Palantir into branch masks into MAGIC into gene
//! trends.
//!
//! The unit tests in each module pin their own maths. This one pins the joins:
//! that the branch column order survives from `terminal_states` through
//! `select_branch_cells` into `GeneTrendsResult::trends`, that a MAGIC operator
//! built on the same kNN graph lines up row for row with the pseudotime, and
//! that a gene planted on one arm of a Y comes back rising on that arm and flat
//! on the other.

#![cfg(feature = "single-cell")]

use faer::Mat;

use bixverse_rs::ml::gp::LandmarkGpParams;
use bixverse_rs::prelude::KnnParams;
use bixverse_rs::single_cell::sc_processing::knn::generate_knn_with_dist;
use bixverse_rs::single_cell::sc_processing::magic::{
    MagicOperator, MagicParams, magic_impute_dense,
};
use bixverse_rs::single_cell::sc_trajectory::gene_trends::{
    BranchSelectionParams, GeneTrendsParams, compute_gene_trends, select_branch_cells,
};
use bixverse_rs::single_cell::sc_trajectory::palantir::{PalantirParams, run_palantir};

/// Cells on the shared trunk of the Y.
///
/// Matches `Y_TRUNK` in the Palantir unit tests. Everything here supplies its
/// terminal states explicitly, which is the case that holds at this size;
/// automatic detection is the one thing that needs the 200-cell fixture.
const TRUNK: usize = 100;
/// Cells per arm.
const ARM: usize = 100;
/// Rate at which the arms separate, per unit of arc.
///
/// Same value the Palantir unit tests use, and load bearing in both directions:
/// too slow and the kNN bridges the arms so the branch never resolves, too fast
/// and they detach from the trunk entirely, which makes the diffusion spectrum
/// degenerate and every downstream number an artefact of the eigensolver.
const ARM_DIVERGENCE: f32 = 1.4;

/// Deterministic jitter in `[-0.5, 0.5)`, so the fixture needs no RNG. `salt`
/// distinguishes independent noise streams.
fn jitter(i: usize, salt: usize) -> f32 {
    let h = (i.wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(40_503)) % 10_007;
    h as f32 / 10_007.0 - 0.5
}

/// A Y-shaped manifold: a shared trunk splitting into two arms.
///
/// Cells `0..TRUNK` are the trunk, the next `ARM` cells are the first branch,
/// the last `ARM` cells the second.
///
/// The second arm is the exact reflection of the first, noise included, which
/// is what makes "the two fates are equivalent" a statement with a ground
/// truth. An independent noise stream per arm makes them genuinely different,
/// and an uneven branch assignment is then a correct answer rather than a bug.
/// Same construction as the Palantir unit tests.
fn y_manifold() -> Vec<Vec<f32>> {
    let mut coords = Vec::with_capacity(TRUNK + 2 * ARM);

    for i in 0..TRUNK {
        let t = i as f32 / TRUNK as f32 * 5.0;
        coords.push(vec![t, jitter(i, 3) * 0.3]);
    }
    for i in 0..ARM {
        let t = (i + 1) as f32 / ARM as f32 * 5.0;
        coords.push(vec![5.0 + t, t * ARM_DIVERGENCE + jitter(i, 4) * 0.3]);
    }
    for i in 0..ARM {
        let t = (i + 1) as f32 / ARM as f32 * 5.0;
        coords.push(vec![5.0 + t, -(t * ARM_DIVERGENCE + jitter(i, 4) * 0.3)]);
    }

    coords
}

/// Exhaustive kNN over the fixture, as `run_palantir` expects it:
/// `(indices, squared distances)`.
fn knn_of(coords: &[Vec<f32>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n = coords.len();
    let d = coords[0].len();
    let mat = Mat::<f32>::from_fn(n, d, |i, j| coords[i][j]);

    let mut params = KnnParams::new();
    params.knn_method = "exhaustive".to_string();
    params.ann_dist = "euclidean".to_string();
    params.k = k;

    let (indices, distances) =
        generate_knn_with_dist(mat.as_ref(), &params, true, false, 42, false).unwrap();
    (indices, distances.unwrap())
}

/// [PalantirParams] with an exact kNN backend, a pinned component count and a
/// pinned spectrum budget. Mirrors `test_params` in the Palantir unit tests.
///
/// The crate default already converges at this size, so the restart budget is
/// insurance: these manifolds' spectra sit within `1e-5` of one across the
/// retained range and the margin closes as the fixture grows, so anyone
/// enlarging it would silently start measuring the eigensolver instead of the
/// data. The `eigen_converged` assertion below is what would catch that.
fn palantir_params() -> PalantirParams {
    let mut params = PalantirParams::new();
    params.knn = 24;
    params.num_waypoints = 120;
    params.knn_params.knn_method = "exhaustive".to_string();
    params.n_eigs = Some(3);
    params.lanczos_params.max_restarts = 256;
    params
}

/// Three planted genes over the Y.
///
/// * Gene 0 switches on along the **first** arm only.
/// * Gene 1 switches on along the **second** arm only.
/// * Gene 2 is a trunk marker that decays once the arms start.
///
/// Counts are noisy and sparse on purpose, so MAGIC has something to do. The
/// result is `n_cells` by 3, log1p scaled.
fn planted_expression() -> Mat<f32> {
    let n = TRUNK + 2 * ARM;

    Mat::<f32>::from_fn(n, 3, |cell, gene| {
        let (arm_a, arm_b) = ((TRUNK..TRUNK + ARM).contains(&cell), cell >= TRUNK + ARM);
        // Position along whichever arm the cell sits on, zero on the trunk.
        let along = if arm_a {
            (cell - TRUNK) as f32 / ARM as f32
        } else if arm_b {
            (cell - TRUNK - ARM) as f32 / ARM as f32
        } else {
            0.0
        };

        let signal = match gene {
            0 if arm_a => along * 6.0,
            1 if arm_b => along * 6.0,
            2 => 6.0 * (1.0 - along),
            _ => 0.0,
        };

        // Dropout plus noise, then log1p, which is the layer trends expect.
        let noise = jitter(cell, 100 + gene);
        let dropped = if jitter(cell, 200 + gene) > 0.2 {
            0.0
        } else {
            1.0
        };
        ((signal + noise).max(0.0) * dropped).ln_1p()
    })
}

/// The whole chain, on a Y with two planted arm-specific genes.
///
/// Terminal states are supplied rather than detected: the detection heuristic
/// is exercised by the Palantir unit tests, and pinning the tips here keeps
/// this test about the joins rather than about that heuristic.
#[test]
fn test_palantir_to_magic_to_gene_trends() {
    let n = TRUNK + 2 * ARM;
    let coords = y_manifold();
    let (knn_indices, knn_distances) = knn_of(&coords, 15);
    let tips = [TRUNK + ARM - 1, TRUNK + 2 * ARM - 1];

    let palantir = run_palantir(
        &knn_indices,
        &knn_distances,
        0,
        Some(&tips),
        palantir_params(),
        42,
        0,
    )
    .expect("palantir runs");

    // Both of these gate everything below. A disconnected graph or an
    // under-resolved embedding makes the branch assignment a property of the
    // eigensolver's budget rather than of the manifold, and the trend
    // assertions would then be measuring the wrong thing while still passing.
    assert!(
        palantir.eigen_converged,
        "the diffusion eigensolve did not converge (residual {:e}); the embedding is under-resolved",
        palantir.eigen_residual
    );
    assert_eq!(
        palantir.repair_edges, 0,
        "the Y fixture is disconnected; every number below would be an eigensolver artefact"
    );
    assert_eq!(palantir.terminal_states, tips);
    assert_eq!(palantir.branch_probs.ncols(), 2);

    // -- branch masks --

    let branch_cells = select_branch_cells(
        &palantir.pseudotime,
        palantir.branch_probs.as_ref(),
        &BranchSelectionParams::default(),
    )
    .expect("branch selection runs");

    assert_eq!(branch_cells.len(), 2);
    for (branch, cells) in branch_cells.iter().enumerate() {
        assert!(!cells.is_empty(), "branch {branch} selected nothing");
        assert!(
            cells.windows(2).all(|w| w[0] < w[1]),
            "branch {branch} is not ascending"
        );
        assert!(cells.iter().all(|&c| c < n));
    }

    // Column order follows `terminal_states`, so branch 0 must own the tip of
    // the first arm and branch 1 the tip of the second, and neither may claim
    // the other's.
    assert!(branch_cells[0].contains(&tips[0]));
    assert!(!branch_cells[0].contains(&tips[1]));
    assert!(branch_cells[1].contains(&tips[1]));
    assert!(!branch_cells[1].contains(&tips[0]));

    // -- MAGIC --

    let cell_indices: Vec<usize> = (0..n).collect();
    let operator = MagicOperator::from_knn(&knn_indices, &knn_distances, &cell_indices, n)
        .expect("operator builds");
    assert_eq!(operator.n_cells(), n);

    let raw = planted_expression();
    // Row-major, which is the layout the dense entry point works in.
    let block: Vec<f32> = (0..n)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| raw[(i, j)])
        .collect();

    let imputed = magic_impute_dense(
        &operator,
        &block,
        3,
        Some(MagicParams {
            clip_threshold: 0.0,
            ..Default::default()
        }),
    )
    .expect("imputation runs");

    // Smoothing must reduce cell-to-cell variance without moving the mean.
    for gene in 0..3 {
        let before: Vec<f32> = (0..n).map(|i| block[i * 3 + gene]).collect();
        let after: Vec<f32> = (0..n).map(|i| imputed[i * 3 + gene]).collect();

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let var = |v: &[f32]| {
            let m = mean(v);
            v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
        };

        assert!(
            var(&after) < var(&before),
            "gene {gene}: diffusion did not smooth ({} -> {})",
            var(&before),
            var(&after)
        );
        assert!(
            (mean(&after) - mean(&before)).abs() < 0.05 * mean(&before).max(1e-3),
            "gene {gene}: the row-stochastic operator moved the mean"
        );
    }

    let expression = Mat::<f32>::from_fn(n, 3, |i, j| imputed[i * 3 + j]);

    // -- gene trends --

    let params = GeneTrendsParams {
        resolution: 200,
        gp: LandmarkGpParams {
            length_scale: 0.15,
            sigma: 0.2,
            ..Default::default()
        },
        ..Default::default()
    };

    let trends = compute_gene_trends(
        expression.as_ref(),
        &palantir.pseudotime,
        &branch_cells,
        None,
        &params,
    )
    .expect("gene trends run");

    assert_eq!(trends.trends.len(), 2);
    for (branch, cells) in branch_cells.iter().enumerate() {
        assert_eq!(trends.trends[branch].nrows(), 200);
        assert_eq!(trends.trends[branch].ncols(), 3);
        assert_eq!(trends.grids[branch].len(), 200);
        assert_eq!(trends.n_cells[branch], cells.len());
        // The grid must be ascending and span the branch's own pseudotime.
        assert!(trends.grids[branch].windows(2).all(|w| w[0] < w[1]));
        assert!(
            trends.trends[branch]
                .col_iter()
                .all(|c| c.iter().all(|v| v.is_finite())),
            "branch {branch} produced a non-finite trend"
        );
    }

    // A gene planted on one arm must end high on its own branch and low on the
    // other. Comparing the last grid point of each branch is the cleanest way
    // to say that, since the two branches share their trunk and only diverge at
    // the end.
    let end = |branch: usize, gene: usize| trends.trends[branch][(199, gene)];

    assert!(
        end(0, 0) > end(1, 0),
        "gene 0 was planted on arm 0 but ends higher on arm 1 ({} vs {})",
        end(0, 0),
        end(1, 0)
    );
    assert!(
        end(1, 1) > end(0, 1),
        "gene 1 was planted on arm 1 but ends higher on arm 0 ({} vs {})",
        end(1, 1),
        end(0, 1)
    );

    // The trunk marker decays on both branches.
    for branch in 0..2 {
        assert!(
            end(branch, 2) < trends.trends[branch][(0, 2)],
            "branch {branch}: the trunk marker did not decay"
        );
    }
}

/// The same joins without the Palantir eigensolve, so CI keeps covering them.
///
/// Pseudotime and fate probabilities are handed in rather than inferred, which
/// removes the only expensive step and the only part that needs a manifold
/// dense enough to stay connected. What survives is everything this file exists
/// to pin: the branch column order flowing through `select_branch_cells`, a
/// MAGIC operator lining up row for row with the pseudotime, and an
/// arm-specific gene coming back on its own branch.
#[test]
fn test_branch_masks_to_magic_to_gene_trends() {
    let n = TRUNK + 2 * ARM;
    let coords = y_manifold();
    let (knn_indices, knn_distances) = knn_of(&coords, 15);

    // Pseudotime straight off the fixture's construction: the trunk runs
    // 0..0.5, each arm 0.5..1.
    let pseudotime: Vec<f32> = (0..n)
        .map(|cell| {
            if cell < TRUNK {
                cell as f32 / TRUNK as f32 * 0.5
            } else if cell < TRUNK + ARM {
                0.5 + (cell - TRUNK) as f32 / ARM as f32 * 0.5
            } else {
                0.5 + (cell - TRUNK - ARM) as f32 / ARM as f32 * 0.5
            }
        })
        .collect();

    // Fate probabilities: even on the trunk, committed on the arms.
    let branch_probs = Mat::<f32>::from_fn(n, 2, |cell, fate| {
        if cell < TRUNK {
            0.5
        } else if cell < TRUNK + ARM {
            if fate == 0 { 1.0 } else { 0.0 }
        } else if fate == 1 {
            1.0
        } else {
            0.0
        }
    });

    let branch_cells = select_branch_cells(
        &pseudotime,
        branch_probs.as_ref(),
        &BranchSelectionParams::default(),
    )
    .expect("branch selection runs");

    assert_eq!(branch_cells.len(), 2);
    // Column order must follow the fate columns, so branch 0 owns the first arm.
    assert!(branch_cells[0].contains(&(TRUNK + ARM - 1)));
    assert!(!branch_cells[0].contains(&(n - 1)));
    assert!(branch_cells[1].contains(&(n - 1)));
    assert!(!branch_cells[1].contains(&(TRUNK + ARM - 1)));
    for cells in &branch_cells {
        assert!(cells.windows(2).all(|w| w[0] < w[1]));
    }

    // -- MAGIC --

    let cell_indices: Vec<usize> = (0..n).collect();
    let operator = MagicOperator::from_knn(&knn_indices, &knn_distances, &cell_indices, n)
        .expect("operator builds");

    let raw = planted_expression();
    let block: Vec<f32> = (0..n)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| raw[(i, j)])
        .collect();

    let imputed = magic_impute_dense(
        &operator,
        &block,
        3,
        Some(MagicParams {
            clip_threshold: 0.0,
            ..Default::default()
        }),
    )
    .expect("imputation runs");

    for gene in 0..3 {
        let before: Vec<f32> = (0..n).map(|i| block[i * 3 + gene]).collect();
        let after: Vec<f32> = (0..n).map(|i| imputed[i * 3 + gene]).collect();

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let var = |v: &[f32]| {
            let m = mean(v);
            v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
        };

        assert!(var(&after) < var(&before), "gene {gene}: no smoothing");
        assert!(
            (mean(&after) - mean(&before)).abs() < 0.05 * mean(&before).max(1e-3),
            "gene {gene}: the row-stochastic operator moved the mean"
        );
    }

    // -- gene trends --

    let expression = Mat::<f32>::from_fn(n, 3, |i, j| imputed[i * 3 + j]);
    let params = GeneTrendsParams {
        resolution: 200,
        gp: LandmarkGpParams {
            length_scale: 0.15,
            sigma: 0.2,
            ..Default::default()
        },
        ..Default::default()
    };

    let trends = compute_gene_trends(
        expression.as_ref(),
        &pseudotime,
        &branch_cells,
        None,
        &params,
    )
    .expect("gene trends run");

    assert_eq!(trends.trends.len(), 2);
    for (branch, cells) in branch_cells.iter().enumerate() {
        assert_eq!(trends.trends[branch].nrows(), 200);
        assert_eq!(trends.trends[branch].ncols(), 3);
        assert_eq!(trends.grids[branch].len(), 200);
        assert_eq!(trends.n_cells[branch], cells.len());
        assert!(trends.grids[branch].windows(2).all(|w| w[0] < w[1]));
        assert!(
            trends.trends[branch]
                .col_iter()
                .all(|c| c.iter().all(|v| v.is_finite()))
        );
    }

    let end = |branch: usize, gene: usize| trends.trends[branch][(199, gene)];
    assert!(
        end(0, 0) > end(1, 0),
        "gene 0 was planted on arm 0 but ends higher on arm 1 ({} vs {})",
        end(0, 0),
        end(1, 0)
    );
    assert!(
        end(1, 1) > end(0, 1),
        "gene 1 was planted on arm 1 but ends higher on arm 0 ({} vs {})",
        end(1, 1),
        end(0, 1)
    );
    for branch in 0..2 {
        assert!(
            end(branch, 2) < trends.trends[branch][(0, 2)],
            "branch {branch}: the trunk marker did not decay"
        );
    }
}
