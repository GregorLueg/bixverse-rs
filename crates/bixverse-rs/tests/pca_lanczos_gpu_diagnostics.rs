//! Block Lanczos PCA on wgpu against the CPU Lanczos and the GPU randomised
//! SVD, on a real store, with the kNN accuracy of every GPU solve against the
//! CPU f64 Lanczos.
//!
//! Accuracy, per embedding width `d` (the leading `d` PCs of both): the mean
//! fraction of each cell's 15 nearest neighbours shared with the reference
//! (a set intersection, order ignored), and the ratio of the mean distance to
//! the candidate's neighbours over the mean distance to the reference's,
//! both measured in the reference embedding. The kNN is KmKnn, the default.
//!
//! Environment:
//!
//! * `BIXVERSE_TIMING_STORE` - directory with `counts_genes.bin`; skipped when
//!   unset
//! * `BIXVERSE_PCA_POOL` - directory with `cells.bin` and `hvg_<n>.bin`,
//!   0-based little-endian i32 indices
//! * `BIXVERSE_PCA_HVG` - comma separated HVG counts, default `2000`
//! * `BIXVERSE_PCA_BLOCKS` - comma separated block sizes, default `16,32,64`
//! * `BIXVERSE_PCA_TOL` - Lanczos tolerance, default the params default
//! * `BIXVERSE_PCA_POWER` - randomised power iterations, default `4`
//! * `BIXVERSE_PCA_OVERSAMPLE` - randomised oversampling, default `20`
//! * `BIXVERSE_PCA_ACC_DIMS` - embedding widths for the accuracy, default
//!   `30,10`
//! * `BIXVERSE_PCA_LANCZOS_ONLY` - if set, skip the CPU and randomised
//!   references and the accuracy, so a profiler run sees only the Lanczos
//!   kernels
//!
//! Prints only, so it sits behind `large_scale_diagnostics`.

#![cfg(all(feature = "sc-gpu", feature = "large_scale_diagnostics"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use cubecl::wgpu::WgpuDevice;
use faer::{Mat, MatRef};
use rayon::prelude::*;

use bixverse_rs::core::math::pca_svd::compute_pc_scores;
use bixverse_rs::core::math::sparse::{BlockLanczosParams, sparse_svd_lanczos};
use bixverse_rs::gpu::linalg::sparse_lanczos_gpu::sparse_svd_lanczos_gpu;
use bixverse_rs::gpu::linalg::sparse_rand_svd_gpu::{RandSvdGpuParams, randomised_sparse_svd_gpu};
use bixverse_rs::single_cell::sc_data::data_io::ParallelSparseReader;
use bixverse_rs::single_cell::sc_processing::knn::generate_knn_kmknn;
use bixverse_rs::single_cell::sc_processing::pca::{
    SingleCellPcaParams, sparse_pca_accelerator_input,
};

/// Principal components, as in the R bench.
const N_PCS: usize = 30;

/// Seed, as in the R bench.
const SEED: u64 = 42;

/// Neighbours for the accuracy, the `params_sc_neighbours()` default.
const K: usize = 15;

/// Read a little-endian i32 index file.
///
/// ### Params
///
/// * `path` - File to read
///
/// ### Returns
///
/// The indices as `usize`.
fn read_indices(path: &Path) -> Vec<usize> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect()
}

/// Comma separated `usize` list from an environment variable.
///
/// ### Params
///
/// * `key` - Variable name
/// * `default` - Value when unset
///
/// ### Returns
///
/// The parsed list.
fn env_list(key: &str, default: &str) -> Vec<usize> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect()
}

/// PC scores as f32, whatever the solver's float.
///
/// ### Params
///
/// * `scores` - Cells x PCs
///
/// ### Returns
///
/// The same matrix in f32.
fn to_f32<T: Into<f64> + Copy>(scores: &Mat<T>) -> Mat<f32> {
    Mat::from_fn(scores.nrows(), scores.ncols(), |i, j| {
        scores[(i, j)].into() as f32
    })
}

/// Reference embedding at one width, with its kNN.
struct Reference {
    /// Leading PCs of the CPU f64 Lanczos.
    scores: Mat<f32>,
    /// Its kNN, self excluded.
    knn: Vec<Vec<usize>>,
    /// Mean distance from each cell to its reference neighbours.
    dist: Vec<f64>,
}

/// Mean Euclidean distance from each cell to the listed neighbours.
///
/// ### Params
///
/// * `embd` - Embedding the distances are measured in
/// * `knn` - Neighbour lists
///
/// ### Returns
///
/// One mean distance per cell.
fn mean_nn_dist(embd: MatRef<f32>, knn: &[Vec<usize>]) -> Vec<f64> {
    knn.par_iter()
        .enumerate()
        .map(|(i, nbrs)| {
            nbrs.iter()
                .map(|&j| {
                    (0..embd.ncols())
                        .map(|c| (embd[(i, c)] as f64 - embd[(j, c)] as f64).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / nbrs.len() as f64
        })
        .collect()
}

/// Build the reference at width `d`.
///
/// ### Params
///
/// * `scores` - CPU f64 Lanczos scores, as f32
/// * `d` - Leading PCs to keep
///
/// ### Returns
///
/// The [`Reference`].
fn reference(scores: &Mat<f32>, d: usize) -> Reference {
    let scores = scores.subcols(0, d).to_owned();
    let knn =
        generate_knn_kmknn(scores.as_ref(), "euclidean", K, None, SEED as usize, false).unwrap();
    let dist = mean_nn_dist(scores.as_ref(), &knn);
    Reference { scores, knn, dist }
}

/// kNN overlap and distance ratio of a candidate against the reference.
///
/// ### Params
///
/// * `reference` - Reference at width `d`
/// * `scores` - Candidate scores, at least `d` columns
///
/// ### Returns
///
/// `(overlap, ratio_mean, ratio_p99)`.
fn accuracy(reference: &Reference, scores: &Mat<f32>) -> (f64, f64, f64) {
    let d = reference.scores.ncols();
    let x = scores.subcols(0, d).to_owned();
    let knn = generate_knn_kmknn(x.as_ref(), "euclidean", K, None, SEED as usize, false).unwrap();

    let overlap = knn
        .par_iter()
        .zip(&reference.knn)
        .map(|(a, b)| a.iter().filter(|j| b.contains(j)).count() as f64 / a.len() as f64)
        .sum::<f64>()
        / knn.len() as f64;

    let mut ratio: Vec<f64> = mean_nn_dist(reference.scores.as_ref(), &knn)
        .iter()
        .zip(&reference.dist)
        .map(|(x, r)| x / r)
        .collect();
    let mean = ratio.iter().sum::<f64>() / ratio.len() as f64;
    ratio.sort_by(f64::total_cmp);
    let p99 = ratio[((ratio.len() as f64 * 0.99) as usize).min(ratio.len() - 1)];

    (overlap, mean, p99)
}

/// Print the accuracy of a candidate at every reference width.
///
/// ### Params
///
/// * `refs` - References, one per width
/// * `scores` - Candidate scores
fn report(refs: &[Reference], scores: &Mat<f32>) {
    for r in refs {
        let (overlap, mean, p99) = accuracy(r, scores);
        println!(
            "    d = {:>2}: knn overlap {overlap:.4}, dist ratio mean {mean:.4} p99 {p99:.4}",
            r.scores.ncols()
        );
    }
}

#[test]
// Real 250k- to 830k-cell subsets, tens of seconds to minutes per HVG set.
fn diagnostics_pca_lanczos_gpu() {
    let Ok(store) = std::env::var("BIXVERSE_TIMING_STORE") else {
        eprintln!("BIXVERSE_TIMING_STORE unset, skipping");
        return;
    };
    let pool = PathBuf::from(std::env::var("BIXVERSE_PCA_POOL").expect("BIXVERSE_PCA_POOL"));
    let hvgs = env_list("BIXVERSE_PCA_HVG", "2000");
    let blocks = env_list("BIXVERSE_PCA_BLOCKS", "16,32,64");
    let powers = env_list("BIXVERSE_PCA_POWER", "4");
    let oversamples = env_list("BIXVERSE_PCA_OVERSAMPLE", "20");
    let dims = env_list("BIXVERSE_PCA_ACC_DIMS", "30,10");
    let tol: f64 = std::env::var("BIXVERSE_PCA_TOL")
        .map(|s| s.parse().unwrap())
        .unwrap_or(BlockLanczosParams::new().tol);
    let lanczos_only = std::env::var("BIXVERSE_PCA_LANCZOS_ONLY").is_ok();

    let reader =
        ParallelSparseReader::new(Path::new(&store).join("counts_genes.bin").to_str().unwrap())
            .unwrap();
    let cells = read_indices(&pool.join("cells.bin"));
    let device = WgpuDevice::default();

    for &h in &hvgs {
        let genes = read_indices(&pool.join(format!("hvg_{h}.bin")));
        let t = Instant::now();
        let (csc, mu, sd) = sparse_pca_accelerator_input(
            &reader,
            &cells,
            &genes,
            None,
            &SingleCellPcaParams::default(),
            "diag",
            0,
        )
        .unwrap();
        println!(
            "\n== {} cells x {} genes, nnz {}, input {:.2?}",
            csc.shape.0,
            csc.shape.1,
            csc.get_nnz(),
            t.elapsed()
        );

        let mu64: Vec<f64> = mu.iter().map(|&x| x as f64).collect();
        let sd64: Vec<f64> = sd.iter().map(|&x| x as f64).collect();
        let (cpu_s, refs) = if lanczos_only {
            (None, Vec::new())
        } else {
            let t = Instant::now();
            let cpu = sparse_svd_lanczos::<f32, f32, f64>(
                &csc,
                N_PCS,
                SEED,
                true,
                Some(&mu64),
                Some(&sd64),
                None,
            )
            .unwrap();
            println!("cpu lanczos (f64):             {:.3?}", t.elapsed());
            let scores = to_f32(&compute_pc_scores(&cpu));
            let refs: Vec<Reference> = dims.iter().map(|&d| reference(&scores, d)).collect();
            (Some(cpu.s), refs)
        };

        // Two runs each: the first pays shader compilation. Accuracy on the
        // second, after its timing.
        for &q in powers.iter().filter(|_| !lanczos_only) {
            for &p in &oversamples {
                for run in 0..2 {
                    let t = Instant::now();
                    let rand = randomised_sparse_svd_gpu::<f32, f32>(
                        csc.clone(),
                        &mu,
                        &sd,
                        None,
                        N_PCS,
                        Some(RandSvdGpuParams::new(q, p)),
                        SEED,
                        device.clone().into(),
                        0,
                    )
                    .unwrap();
                    if run == 1 {
                        println!("gpu randomised q = {q} p = {p:>2}:  {:.3?}", t.elapsed());
                        report(&refs, &to_f32(&compute_pc_scores(&rand)));
                    }
                }
            }
        }

        for &b in &blocks {
            let params = BlockLanczosParams {
                block_size: b,
                tol,
                ..BlockLanczosParams::new()
            };
            for run in 0..2 {
                let t = Instant::now();
                let svd = sparse_svd_lanczos_gpu::<f32>(
                    csc.clone(),
                    &mu,
                    &sd,
                    None,
                    N_PCS,
                    Some(params),
                    SEED,
                    device.clone().into(),
                    if run == 1 { 2 } else { 0 },
                )
                .unwrap();
                if run == 1 {
                    println!("gpu lanczos b = {b:>2}:            {:.3?}", t.elapsed());
                    if let Some(cpu_s) = &cpu_s {
                        let worst = (0..N_PCS)
                            .map(|j| (svd.s[j] as f64 - cpu_s[j]).abs() / cpu_s[j])
                            .fold(0.0f64, f64::max);
                        println!("    worst rel s error vs cpu: {worst:.2e}");
                    }
                    report(&refs, &to_f32(&compute_pc_scores(&svd)));
                }
            }
        }
    }
}
