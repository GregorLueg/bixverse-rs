//! SEACells end to end on MLX and (with `gpu`) cubecl, with the B-gradient
//! argmin timed per call inside the same fits.
//!
//! Every back end runs behind a timing wrapper around the [FwArgminB] seam, so
//! the per-call argmin numbers (upload, kernels and readback) come from the
//! real Frank-Wolfe trajectory rather than from a synthetic operand: `nnz(K²B)`
//! and the density of `t1` both grow over a fit, and the kernel cost follows
//! them. No CPU arm: at 200k cells it takes far longer than both GPU fits.
//!
//! Run with:
//! ```text
//! cargo bench --features mlx,single-cell --bench mlx_seacells_bench
//! cargo bench --features mlx,single-cell,gpu --bench mlx_seacells_bench
//! ```
//!
//! Environment: `BIXVERSE_BENCH_SMALL=1` skips the 200k shape.
//!
//! Needs `libmlxc` (see `build.rs`); without it the `mlx` module does not
//! exist and this does not compile.

use std::time::{Duration, Instant};

use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use bixverse_rs::mlx::sc::seacells::MlxFwArgminB;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_generation::seacells::{
    FwArgminB, FwAtoms, SEACells, SEACellsParams,
};
use bixverse_rs::single_cell::sc_processing::knn::{KnnParams, generate_knn_with_dist};

///////////////
// Constants //
///////////////

/// Seed for the kNN graph, the archetype initialisation and the fits.
const SEED: usize = 42;

/// Embedding width.
const EMBED_DIM: usize = 30;

/// Seed of the synthetic embedding.
const EMBED_SEED: u64 = 7;

/// Blobs in the synthetic embedding.
const N_CLUSTERS: usize = 40;

/// `(cells, SEACells, kNN)` shapes, the second only without
/// `BIXVERSE_BENCH_SMALL`.
const SHAPES: [(usize, usize, usize); 2] = [(50_000, 666, 15), (200_000, 2_666, 15)];

/////////////
// Helpers //
/////////////

/// Clustered synthetic embedding, the same as `seacells_gpu_bench`.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `dim` - Embedding width
/// * `seed` - Random seed
///
/// ### Returns
///
/// An `n x dim` matrix.
fn make_embedding(n: usize, dim: usize, seed: u64) -> Mat<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centre_dist = Normal::new(0.0f32, 1.0).expect("valid normal");
    let noise_dist = Normal::new(0.0f32, 0.55).expect("valid normal");

    let centres: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..dim).map(|_| centre_dist.sample(&mut rng)).collect())
        .collect();
    let assignments: Vec<usize> = (0..n).map(|_| rng.random_range(0..N_CLUSTERS)).collect();

    Mat::from_fn(n, dim, |i, j| {
        centres[assignments[i]][j] + noise_dist.sample(&mut rng)
    })
}

/// Parameters of the end-to-end runs, as in `seacells_gpu_bench`.
///
/// ### Params
///
/// * `n_sea_cells` - Number of SEACells
/// * `knn` - Neighbours in the kNN graph
///
/// ### Returns
///
/// The populated [SEACellsParams].
fn e2e_params(n_sea_cells: usize, knn: usize) -> SEACellsParams {
    let mut knn_params = KnnParams::new();
    knn_params.k = knn;
    knn_params.ann_dist = "euclidean".to_string();

    SEACellsParams {
        n_sea_cells,
        max_fw_iters: 50,
        convergence_epsilon: 1e-3,
        max_iter: 3,
        min_iter: 3,
        greedy_threshold: 0,
        graph_building: "union".to_string(),
        pruning: true,
        pruning_threshold: 1e-7,
        n_landmarks: None,
        knn_params,
        lanczos_params: LanczosParams::default(),
    }
}

/// A back end with every seam call timed.
struct Timed<B> {
    /// The wrapped back end
    inner: B,
    /// Time in `begin`
    begin: Duration,
    /// Time in `argmins`
    argmins: Duration,
    /// Number of `argmins` calls
    calls: usize,
    /// Time in `columns_a`
    columns_a: Duration,
}

impl<B> Timed<B> {
    /// Wrap a back end.
    ///
    /// ### Params
    ///
    /// * `inner` - Back end to time
    ///
    /// ### Returns
    ///
    /// The wrapper with zeroed timers.
    fn new(inner: B) -> Self {
        Self {
            inner,
            begin: Duration::ZERO,
            argmins: Duration::ZERO,
            calls: 0,
            columns_a: Duration::ZERO,
        }
    }
}

impl<B: FwArgminB> FwArgminB for Timed<B> {
    fn begin(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        t2: &CompressedSparseData2<f32>,
    ) -> Result<(), BixverseErrors> {
        let start = Instant::now();
        let out = self.inner.begin(t1, t2);
        self.begin += start.elapsed();
        out
    }

    fn argmins(
        &mut self,
        k2_b: &CompressedSparseData2<f32>,
        b: &CompressedSparseData2<f32>,
    ) -> Result<(Vec<usize>, f32), BixverseErrors> {
        let start = Instant::now();
        let out = self.inner.argmins(k2_b, b);
        self.argmins += start.elapsed();
        self.calls += 1;
        out
    }

    fn columns_a(
        &mut self,
        t1: &CompressedSparseData2<f32>,
        a_prev_t: &CompressedSparseData2<f32>,
        k2_b: &CompressedSparseData2<f32>,
        k: usize,
        n: usize,
        n_iters: usize,
        pruning: Option<f32>,
    ) -> Result<Vec<FwAtoms>, BixverseErrors> {
        let start = Instant::now();
        let out = self
            .inner
            .columns_a(t1, a_prev_t, k2_b, k, n, n_iters, pruning);
        self.columns_a += start.elapsed();
        out
    }
}

/// One timed fit: model set-up plus [SEACells::fit_with] on `backend`.
///
/// ### Params
///
/// * `name` - Label for the output row
/// * `embedding` - Cell embedding
/// * `knn` - kNN indices and distances
/// * `params` - Fit parameters
/// * `make_backend` - Builds the back end once `k` is known
///
/// ### Returns
///
/// The hard assignments.
fn run_fit<B: FwArgminB>(
    name: &str,
    embedding: &Mat<f32>,
    knn: &(Vec<Vec<usize>>, Vec<Vec<f32>>),
    params: &SEACellsParams,
    make_backend: impl FnOnce(usize, usize) -> B,
) -> Vec<usize> {
    let n = embedding.nrows();
    let start = Instant::now();
    let mut model = SEACells::new(n, params);
    model.construct_kernel_mat(embedding.as_ref(), &knn.0, &knn.1, 0);
    model
        .initialise_archetypes(&knn.0, &knn.1, 0, SEED as u64)
        .expect("archetype init failed");
    let k = model.get_archetypes().expect("no archetypes").len();
    let mut backend = Timed::new(make_backend(n, k));
    model.fit_with(SEED, 0, &mut backend).expect("fit failed");
    let total = start.elapsed();
    let rss = model.get_rss_history();

    println!(
        "  {:<8} total {:>7.2} s | argmin {:>7.2} s over {:>3} calls = {:>7.2} ms/call | \
         begin {:>5.2} s | columns_a {:>5.2} s | RSS {:.4} -> {:.4}",
        name,
        total.as_secs_f64(),
        backend.argmins.as_secs_f64(),
        backend.calls,
        backend.argmins.as_secs_f64() * 1e3 / backend.calls.max(1) as f64,
        backend.begin.as_secs_f64(),
        backend.columns_a.as_secs_f64(),
        rss[0],
        rss[rss.len() - 1],
    );
    model.get_hard_assignments().expect("assignments failed")
}

//////////
// Main //
//////////

fn main() {
    let small = std::env::var("BIXVERSE_BENCH_SMALL").is_ok();
    println!("\nSEACells end to end, 3 outer iterations x 50 FW steps, pruning 1e-7\n");

    for (i, &(n, k, knn)) in SHAPES.iter().enumerate() {
        if small && i > 0 {
            break;
        }
        let params = e2e_params(k, knn);
        let embedding = make_embedding(n, EMBED_DIM, EMBED_SEED);
        let (knn_indices, knn_distances) = generate_knn_with_dist(
            embedding.as_ref(),
            &params.knn_params,
            true,
            false,
            SEED,
            false,
        )
        .expect("kNN failed");
        let knn = (knn_indices, knn_distances.expect("distances requested"));

        println!("n = {n}, k = {k}");
        let mlx_assign = run_fit("mlx", &embedding, &knn, &params, |n, k| {
            MlxFwArgminB::new(n, k).expect("MLX back end")
        });

        #[cfg(feature = "gpu")]
        {
            use bixverse_rs::gpu::sc_gpu::seacells_gpu::GpuFwArgminB;
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

            let gpu_assign = run_fit("cubecl", &embedding, &knn, &params, |n, k| {
                let client = <WgpuRuntime as cubecl::Runtime>::client(&WgpuDevice::DefaultDevice);
                GpuFwArgminB::<WgpuRuntime>::new(n, k, client).expect("cubecl back end")
            });
            let agree = mlx_assign
                .iter()
                .zip(&gpu_assign)
                .filter(|(a, b)| a == b)
                .count();
            println!(
                "  assignments agree {:.1}%",
                100.0 * agree as f64 / n as f64
            );
        }
        #[cfg(not(feature = "gpu"))]
        let _ = mlx_assign;
        println!();
    }
}
