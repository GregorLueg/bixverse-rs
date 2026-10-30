//! Peak memory and wall time of building the HVG CSR the sparse PCA solvers
//! consume, two ways:
//!
//! * `transpose` - `read_genes_sparse` over all HVGs, then the single-layer
//!   CSC to CSR transpose. Holds both orientations at once.
//! * `csr` - `read_genes_sparse_csr`, the batched read that never holds the
//!   CSC next to the CSR.
//! * `mlx_lanczos` / `mlx_randomised` (feature `mlx`) - the whole
//!   `pca_on_sc_sparse_mlx` with that solver, for its host-side peak.
//!
//! Memory is the macOS physical footprint (anonymous plus compressed, no clean
//! mmap pages), sampled every [`SAMPLE_MS`] ms by a background thread. Run one
//! variant per process so earlier stages do not leak into the allocator state.
//! Both variants print a fingerprint of their CSR; they must agree.
//!
//! Run with:
//! ```text
//! PCA_INPUT_BENCH_GENES=/path/to/genes.bin PCA_INPUT_BENCH_VARIANT=csr \
//!   cargo bench -p bixverse-rs --features single-cell --bench pca_input_bench
//! ```
//!
//! Environment:
//!
//! * `PCA_INPUT_BENCH_GENES` - the gene-major store
//! * `PCA_INPUT_BENCH_VARIANT` - `transpose`, `csr`, `mlx_lanczos` or
//!   `mlx_randomised`
//! * `PCA_INPUT_BENCH_HVG` - number of HVGs, default 2000. Selected once by
//!   VST and cached next to the store
//! * `RAYON_NUM_THREADS` - thread count

#![cfg(feature = "single-cell")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indexmap::IndexSet;
use rayon::prelude::*;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_data::data_io::{
    DataLayerReturn, ParallelSparseReader, SingleCellReading,
};
use bixverse_rs::single_cell::sc_processing::hvg::get_hvg_vst;

/// Sampling interval of the footprint thread.
const SAMPLE_MS: u64 = 10;

/// Components computed by the MLX variants.
#[cfg(feature = "mlx")]
const MLX_PCS: usize = 30;

/// Highest footprint seen since the current stage started.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Current physical footprint of this process.
///
/// ### Returns
///
/// Bytes, or `0` off macOS.
fn footprint() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        );
        info.ri_phys_footprint
    }
    #[cfg(not(target_os = "macos"))]
    0
}

/// Run one stage and print its wall time, peak and settled footprint.
///
/// ### Params
///
/// * `name` - Stage label
/// * `f` - The stage
///
/// ### Returns
///
/// Whatever the stage returns.
fn stage<R>(name: &str, f: impl FnOnce() -> R) -> R {
    let before = footprint();
    PEAK.store(before, Ordering::Relaxed);
    let t = Instant::now();
    let r = f();
    let wall = t.elapsed();
    let after = footprint();
    let peak = PEAK.load(Ordering::Relaxed).max(after);
    println!(
        "{name:<10} {:>8.2} s   before {:>6.2} GB   peak {:>6.2} GB   after {:>6.2} GB",
        wall.as_secs_f64(),
        before as f64 / 1e9,
        peak as f64 / 1e9,
        after as f64 / 1e9
    );
    r
}

/// Order-independent summary of a CSR, for cross-checking the variants.
///
/// ### Params
///
/// * `csr` - The matrix
///
/// ### Returns
///
/// Printable fingerprint.
fn fingerprint(csr: &CompressedSparseData2<f32>) -> String {
    let vals = csr.data_2.as_ref().expect("norm layer");
    let idx_sum: u64 = csr.indices.par_iter().map(|&i| i as u64).sum();
    let val_sum: f64 = vals.par_iter().map(|&v| v as f64).sum();
    let ptr_sum: u64 = csr.indptr.par_iter().map(|&p| p as u64).sum();
    format!(
        "{:?} {:?} nnz {} idx {} ptr {} val {:.6e}",
        csr.cs_type,
        csr.shape,
        csr.indices.len(),
        idx_sum,
        ptr_sum,
        val_sum
    )
}

/// HVGs by VST, cached next to the store, ascending.
///
/// ### Params
///
/// * `path` - The gene store path, the cache sits next to it
/// * `genes` - The gene-major reader
/// * `cells` - Cells to use
/// * `n_hvg` - How many to keep
///
/// ### Returns
///
/// The HVG indices, sorted ascending.
fn hvgs(path: &str, genes: &ParallelSparseReader, cells: &[usize], n_hvg: usize) -> Vec<usize> {
    let cache = format!("{path}.hvg{n_hvg}.txt");
    if let Ok(s) = std::fs::read_to_string(&cache) {
        return s
            .lines()
            .map(|l| l.parse().expect("cached index"))
            .collect();
    }
    let res = get_hvg_vst(genes, cells, 0.3, None, 0).expect("hvg");
    let mut order: Vec<usize> = (0..res.var_std.len()).collect();
    order.sort_unstable_by(|&a, &b| res.var_std[b].total_cmp(&res.var_std[a]));
    let mut hvg = order[..n_hvg].to_vec();
    hvg.sort_unstable();
    let text: Vec<String> = hvg.iter().map(|g| g.to_string()).collect();
    std::fs::write(&cache, text.join("\n")).expect("write hvg cache");
    hvg
}

fn main() {
    let path = std::env::var("PCA_INPUT_BENCH_GENES").expect("set PCA_INPUT_BENCH_GENES");
    let variant = std::env::var("PCA_INPUT_BENCH_VARIANT").unwrap_or_else(|_| "csr".into());
    let n_hvg: usize = std::env::var("PCA_INPUT_BENCH_HVG")
        .ok()
        .map_or(2000, |s| s.parse().expect("PCA_INPUT_BENCH_HVG"));

    std::thread::spawn(|| {
        loop {
            PEAK.fetch_max(footprint(), Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(SAMPLE_MS));
        }
    });

    let genes = ParallelSparseReader::new(&path).expect("gene file");
    let n_cells = genes.get_header().total_cells;
    let cells: Vec<usize> = (0..n_cells).collect();
    let hvg = hvgs(&path, &genes, &cells, n_hvg);
    let cell_set: IndexSet<u32> = cells.iter().map(|&c| c as u32).collect();

    println!(
        "{variant}: {n_cells} cells, {} HVGs, {} threads",
        hvg.len(),
        rayon::current_num_threads()
    );

    let csr = match variant.as_str() {
        "transpose" => {
            let csc = stage("read", || {
                genes
                    .read_genes_sparse::<f32>(&hvg, Some(&cell_set), &DataLayerReturn::Norm)
                    .expect("read")
            });
            stage("transpose", || {
                let csr = csc.transform_single_layer(true).expect("transpose");
                drop(csc);
                csr
            })
        }
        "csr" => stage("read_csr", || {
            genes
                .read_genes_sparse_csr::<f32>(&hvg, Some(&cell_set), &DataLayerReturn::Norm)
                .expect("read")
        }),
        #[cfg(feature = "mlx")]
        "mlx_lanczos" | "mlx_randomised" => {
            use bixverse_rs::mlx::sc::pca::pca_on_sc_sparse_mlx;
            use bixverse_rs::single_cell::sc_processing::pca::{PcaSolver, SingleCellPcaParams};

            let solver = if variant == "mlx_lanczos" {
                PcaSolver::Exact
            } else {
                PcaSolver::Randomised
            };
            let params = SingleCellPcaParams {
                svd_solver: solver,
                ..SingleCellPcaParams::default()
            };
            let (scores, _, s) = stage("mlx_pca", || {
                pca_on_sc_sparse_mlx(&genes, &cells, &hvg, MLX_PCS, None, &params, 42, 0)
                    .expect("mlx pca")
            });
            println!("scores {:?}, s[0] {:.6e}", scores.shape(), s[0]);
            return;
        }
        other => panic!("unknown variant {other}"),
    };

    println!("{}", fingerprint(&csr));
}
