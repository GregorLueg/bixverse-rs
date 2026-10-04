//! End-to-end Harmony benchmark on a real PCA embedding: v1, v2 and (with
//! `gpu`) GPU v2.
//!
//! Input is a raw little-endian `f32` embedding (`{name}_pcs.f32`, row-major
//! N x d) and `u32` batch labels (`{name}_labels.u32`) in `BIXVERSE_HARMONY_DIR`.
//! The first run of each method is at detailed verbosity and prints the stage
//! split; the remaining runs are silent and timed. Each method's corrected
//! embedding is written to `{dir}/{name}_{method}_out.f32` for parity checks.
//!
//! Environment:
//!
//! - `BIXVERSE_HARMONY_DIR` (required) data directory
//! - `BIXVERSE_HARMONY_NAME` dataset name, default `ircolitis`
//! - `BIXVERSE_HARMONY_TILE` tile the data this many times with small jitter
//! - `BIXVERSE_HARMONY_REPS` silent timed runs per method, default 3
//! - `BIXVERSE_HARMONY_ONLY` one of `v1`, `v2`, `gpu`
//! - `BIXVERSE_HARMONY_KM_ITERS` override the CPU initial k-means iterations
//!
//! ```bash
//! BIXVERSE_HARMONY_DIR=/path cargo bench --features single-cell --bench harmony_bench
//! ```

use std::time::{Duration, Instant};

use faer::{Mat, MatRef};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use bixverse_rs::ml::clustering::k_means::KMeansParamsWrappers;
use bixverse_rs::single_cell::sc_batch_correction::harmony::{HarmonyParams, harmony};
use bixverse_rs::single_cell::sc_batch_correction::harmony_v2::{HarmonyParamsV2, harmony_v2};

/// Jitter added to tiled copies, in PC units.
const TILE_JITTER: f32 = 0.05;

/// Read a raw little-endian file of 4-byte values.
///
/// ### Params
///
/// * `path` - File path
///
/// ### Returns
///
/// The raw 4-byte words
fn read_words(path: &str) -> Vec<[u8; 4]> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// Load the embedding and labels, optionally tiled with jitter.
///
/// ### Params
///
/// * `dir` - Data directory
/// * `name` - Dataset name
/// * `tile` - Number of copies
///
/// ### Returns
///
/// `(embedding N x d, labels)`
fn load(dir: &str, name: &str, tile: usize) -> (Mat<f32>, Vec<usize>) {
    let x: Vec<f32> = read_words(&format!("{dir}/{name}_pcs.f32"))
        .into_iter()
        .map(f32::from_le_bytes)
        .collect();
    let labels: Vec<usize> = read_words(&format!("{dir}/{name}_labels.u32"))
        .into_iter()
        .map(|w| u32::from_le_bytes(w) as usize)
        .collect();
    let n = labels.len();
    let d = x.len() / n;

    let mut rng = StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0f32, TILE_JITTER).unwrap();
    let mut out = Mat::<f32>::zeros(n * tile, d);
    for t in 0..tile {
        for i in 0..n {
            for j in 0..d {
                let jitter = if t == 0 { 0.0 } else { noise.sample(&mut rng) };
                out[(t * n + i, j)] = x[i * d + j] + jitter;
            }
        }
    }
    let labels = (0..tile).flat_map(|_| labels.iter().copied()).collect();
    (out, labels)
}

/// Write an N x d matrix as row-major little-endian `f32`.
///
/// ### Params
///
/// * `path` - Output path
/// * `m` - Matrix
fn write_mat(path: &str, m: MatRef<f32>) {
    let mut bytes = Vec::with_capacity(m.nrows() * m.ncols() * 4);
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            bytes.extend_from_slice(&m[(i, j)].to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

/// Run one verbose pass (stage split) then `reps` silent timed passes.
///
/// ### Params
///
/// * `label` - Method label
/// * `reps` - Silent timed runs
/// * `out_path` - Where the corrected embedding goes
/// * `f` - Runs the method at the given verbosity
fn bench(label: &str, reps: usize, out_path: &str, mut f: impl FnMut(usize) -> Mat<f32>) {
    println!("\n##### {label} #####");
    let t = Instant::now();
    let z = f(2);
    println!("{label}: verbose run {:.3} s", t.elapsed().as_secs_f64());
    write_mat(out_path, z.as_ref());

    let mut times: Vec<Duration> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f(0));
        times.push(t.elapsed());
    }
    if let (Some(best), Some(worst)) = (times.iter().min(), times.iter().max()) {
        println!(
            "{label}: best {:.3} s, worst {:.3} s over {reps} runs",
            best.as_secs_f64(),
            worst.as_secs_f64()
        );
    }
}

fn main() {
    let dir = std::env::var("BIXVERSE_HARMONY_DIR").expect("set BIXVERSE_HARMONY_DIR");
    let name = std::env::var("BIXVERSE_HARMONY_NAME").unwrap_or_else(|_| "ircolitis".into());
    let tile: usize = std::env::var("BIXVERSE_HARMONY_TILE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let reps: usize = std::env::var("BIXVERSE_HARMONY_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let only = std::env::var("BIXVERSE_HARMONY_ONLY").ok();
    let km_iters: Option<usize> = std::env::var("BIXVERSE_HARMONY_KM_ITERS")
        .ok()
        .and_then(|v| v.parse().ok());
    let run = |m: &str| only.as_deref().is_none_or(|o| o == m);

    let uptime = std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("uptime: {uptime}");

    let (pca, labels) = load(&dir, &name, tile);
    println!(
        "{name} x{tile}: {} cells, {} dims, {} levels",
        pca.nrows(),
        pca.ncols(),
        labels.iter().max().map_or(0, |m| m + 1)
    );
    let batch = vec![labels];
    let tag = format!("{dir}/{name}x{tile}");

    if run("v1") {
        let mut params = HarmonyParams::default();
        if let Some(it) = km_iters {
            params.kmeans_params = KMeansParamsWrappers::new(it, None, None);
        }
        bench("v1", reps, &format!("{tag}_v1_out.f32"), |v| {
            harmony(pca.as_ref(), &batch, &params, 42, v).unwrap()
        });
    }

    if run("v2") {
        let mut params = HarmonyParamsV2::default();
        if let Some(it) = km_iters {
            params.kmeans_params = KMeansParamsWrappers::new(it, None, None);
        }
        bench("v2", reps, &format!("{tag}_v2_out.f32"), |v| {
            harmony_v2(pca.as_ref(), &batch, &params, 42, v).unwrap()
        });
    }

    #[cfg(feature = "gpu")]
    if run("gpu") {
        use bixverse_rs::gpu::sc_gpu::harmony_gpu::{HarmonyParamsV2Gpu, harmony_v2_gpu};
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        let params = HarmonyParamsV2Gpu::default();
        bench("gpu", reps, &format!("{tag}_gpu_out.f32"), |v| {
            harmony_v2_gpu::<WgpuRuntime>(
                pca.as_ref(),
                &batch,
                &params,
                42,
                WgpuDevice::default(),
                v,
            )
            .unwrap()
        });
    }
}
