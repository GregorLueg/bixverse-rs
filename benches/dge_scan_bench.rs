//! Where the one-vs-rest / one-vs-many DGE wall clock goes.
//!
//! The store is a gamma-Poisson draw calibrated like `hvg_bench`, written cell
//! based and then transposed with `write_gene_file`, exactly as the R side
//! builds its two stores. Cells are split into `DGE_BENCH_GROUPS` groups, each
//! with a small block of up-regulated marker genes.
//!
//! The measured cells:
//!
//! * `read_cells` - decode every cell from the cell store. One arm of the
//!   current per-group loop pays at least this.
//! * `read_genes` - decode every gene from the gene store in blocks. The floor
//!   of a gene-wise scan.
//! * `arm_cells` - one `calculate_dge_grps_mann_whitney` call, group 0 against
//!   the rest. `find_all_markers_sc` pays this once per group.
//! * `scan_rest` - `calculate_dge_one_vs_rest_mann_whitney` over all groups,
//!   one pass over the gene store.
//! * `scan_many` - `calculate_dge_one_vs_many_auroc` with every group as a
//!   reference arm, one pass over the gene store.
//!
//! Warm page cache throughout: every cell runs after a warm-up read.
//!
//! Run with:
//! ```
//! cargo bench --features single-cell --bench dge_scan_bench
//! ```
//!
//! `DGE_BENCH_GENES` / `DGE_BENCH_CELLS` / `DGE_BENCH_GROUPS` change the shape,
//! `DGE_BENCH_REGEN=1` rebuilds the cached stores.

#![cfg(feature = "single-cell")]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};
use rayon::prelude::*;

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::dge_pathway_scores::{
    calculate_dge_grps_mann_whitney, calculate_dge_one_vs_many_auroc,
    calculate_dge_one_vs_rest_mann_whitney,
};
use bixverse_rs::single_cell::sc_data::data_io::CellGeneSparseWriter;
use bixverse_rs::single_cell::sc_data::gene_file_io::write_gene_file;

////////////
// Shapes //
////////////

/// Default gene count. Roughly a filtered 10x feature set.
const DEFAULT_GENES: usize = 20_000;

/// Default cell count.
const DEFAULT_CELLS: usize = 20_000;

/// Default number of cell groups.
const DEFAULT_GROUPS: usize = 10;

/// Genes read per `read_gene_parallel` call in `read_genes`.
const GENE_BLOCK: usize = 1_000;

/// Cells generated per write block.
const WRITE_BLOCK: usize = 2_048;

/// Repeats per timed phase. The fastest run is reported.
const REPEATS: usize = 3;

/// Library size the `data_norm` layer is scaled against.
const TARGET_SIZE: f32 = 1e4;

/// Seed for the generator. Fixed so the cached stores are reproducible.
const SEED: u64 = 0xD6E_5CA4_u64;

/////////////////
// Count model //
/////////////////

/// Median library size, i.e. total UMIs per cell.
const MEDIAN_LIBRARY_SIZE: f64 = 5_000.0;

/// Log-scale spread of the library size.
const LIBRARY_SIZE_LOG_SD: f64 = 0.35;

/// Log-scale spread of per-gene relative expression, before normalisation.
const EXPRESSION_LOG_SD: f64 = 2.0;

/// Biological coefficient of variation of the gamma mixing weight.
const BCV: f64 = 0.55;

/// Below this expected count the gamma mixing weight is skipped.
const OVERDISPERSION_CUTOFF: f64 = 0.25;

/// Poisson draws at or above this rate go through the rejection sampler.
const POISSON_INVERSION_LIMIT: f64 = 30.0;

/// Marker genes per group.
const MARKERS_PER_GROUP: usize = 20;

/// Fold change of a group's marker genes inside the group.
const MARKER_FOLD: f64 = 4.0;

//////////////////////
// Store generation //
//////////////////////

/// Shape of the bench run.
struct Shape {
    /// Number of genes
    n_genes: usize,
    /// Number of cells
    n_cells: usize,
    /// Number of cell groups
    n_groups: usize,
}

/// Read a `usize` from the environment, falling back to a default.
///
/// ### Params
///
/// * `key` - Variable name
/// * `default` - Fallback
///
/// ### Returns
///
/// The parsed value or the default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Group of a cell. Seeded hash so groups are interleaved across the store.
///
/// ### Params
///
/// * `cell` - Cell index
/// * `n_groups` - Number of groups
///
/// ### Returns
///
/// The group index.
fn group_of(cell: usize, n_groups: usize) -> usize {
    let h = (cell as u64 ^ SEED).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ((h >> 32) % n_groups as u64) as usize
}

/// Draw a Poisson count.
///
/// ### Params
///
/// * `rng` - Thread-local generator
/// * `lambda` - The rate
///
/// ### Returns
///
/// The count, or `0` for a non-positive or non-finite rate.
#[inline]
fn poisson_draw(rng: &mut SmallRng, lambda: f64) -> u32 {
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }

    if lambda < POISSON_INVERSION_LIMIT {
        let u: f64 = rng.random();
        let mut p = (-lambda).exp();
        let mut cumulative = p;
        let mut k = 0u32;
        while u > cumulative && k < 10_000 {
            k += 1;
            p *= lambda / k as f64;
            cumulative += p;
        }
        k
    } else {
        Poisson::new(lambda).map_or(0, |d| d.sample(rng) as u32)
    }
}

/// Generate one cell.
///
/// ### Params
///
/// * `cell` - Cell index, also seeds the generator
/// * `gene_share` - Per-gene expression share, sums to one
/// * `library` - Expected library size of this cell
/// * `n_groups` - Number of groups
///
/// ### Returns
///
/// The `CsrCellChunk`, normalised to [`TARGET_SIZE`].
fn draw_cell(cell: usize, gene_share: &[f64], library: f64, n_groups: usize) -> CsrCellChunk {
    let mut rng = SmallRng::seed_from_u64(SEED ^ (cell as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    let phi = BCV * BCV;
    let gamma = Gamma::new(1.0 / phi, phi).expect("valid gamma");

    let group = group_of(cell, n_groups);
    let markers = group * MARKERS_PER_GROUP..(group + 1) * MARKERS_PER_GROUP;

    let mut raw: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (gene, &share) in gene_share.iter().enumerate() {
        let fold = if markers.contains(&gene) {
            MARKER_FOLD
        } else {
            1.0
        };
        let mean = share * library * fold;
        let weight = if mean > OVERDISPERSION_CUTOFF {
            gamma.sample(&mut rng)
        } else {
            1.0
        };
        let count = poisson_draw(&mut rng, mean * weight);
        if count > 0 {
            raw.push(count);
            indices.push(gene as u32);
        }
    }

    CsrCellChunk::from_data(&raw, &indices, cell, TARGET_SIZE, true)
}

/// Paths of the cached cell and gene stores for a shape.
///
/// ### Params
///
/// * `shape` - The bench shape
///
/// ### Returns
///
/// `(cell store, gene store)`.
fn store_paths(shape: &Shape) -> (PathBuf, PathBuf) {
    let stem = format!(
        "bixverse_dge_bench_{}x{}_g{}_{SEED:x}",
        shape.n_genes, shape.n_cells, shape.n_groups
    );
    let dir = std::env::temp_dir();
    (
        dir.join(format!("{stem}_cells.bin")),
        dir.join(format!("{stem}_genes.bin")),
    )
}

/// Write both stores unless matching ones are cached.
///
/// ### Params
///
/// * `shape` - The bench shape
///
/// ### Returns
///
/// `(cell store, gene store)`.
fn ensure_stores(shape: &Shape) -> (PathBuf, PathBuf) {
    let (cell_path, gene_path) = store_paths(shape);
    let regenerate = std::env::var("DGE_BENCH_REGEN").is_ok();

    if cell_path.exists() && gene_path.exists() && !regenerate {
        println!("store: reusing {}", cell_path.display());
        return (cell_path, gene_path);
    }

    println!(
        "store: generating {} genes x {} cells, {} groups",
        shape.n_genes, shape.n_cells, shape.n_groups
    );
    let start = Instant::now();

    let mut rng = SmallRng::seed_from_u64(SEED);
    let shares = LogNormal::new(0.0, EXPRESSION_LOG_SD).expect("valid log-normal");
    let mut gene_share: Vec<f64> = (0..shape.n_genes)
        .map(|_| shares.sample(&mut rng))
        .collect();
    let total: f64 = gene_share.iter().sum();
    gene_share.iter_mut().for_each(|s| *s /= total);

    let libraries =
        LogNormal::new(MEDIAN_LIBRARY_SIZE.ln(), LIBRARY_SIZE_LOG_SD).expect("valid log-normal");
    let library_size: Vec<f64> = (0..shape.n_cells)
        .map(|_| libraries.sample(&mut rng))
        .collect();

    let mut writer =
        CellGeneSparseWriter::new(&cell_path, true, shape.n_cells, shape.n_genes, TARGET_SIZE)
            .expect("writer");

    for block_start in (0..shape.n_cells).step_by(WRITE_BLOCK) {
        let block_end = (block_start + WRITE_BLOCK).min(shape.n_cells);
        let chunks: Vec<CsrCellChunk> = (block_start..block_end)
            .into_par_iter()
            .map(|cell| draw_cell(cell, &gene_share, library_size[cell], shape.n_groups))
            .collect();
        for chunk in chunks {
            writer.write_cell_chunk(chunk).expect("write cell chunk");
        }
    }
    writer.finalise().expect("finalise");

    write_gene_file(
        cell_path.to_str().expect("utf-8 path"),
        gene_path.to_str().expect("utf-8 path"),
        None,
        false,
    )
    .expect("gene file");

    println!("store: written in {:.2?}", start.elapsed());

    (cell_path, gene_path)
}

////////////
// Timing //
////////////

/// Run a closure [`REPEATS`] times and return the fastest wall time.
///
/// ### Params
///
/// * `f` - The work to time
///
/// ### Returns
///
/// The fastest run.
fn best_of<F: FnMut()>(mut f: F) -> Duration {
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .expect("REPEATS > 0")
}

//////////
// Main //
//////////

fn main() {
    let shape = Shape {
        n_genes: env_usize("DGE_BENCH_GENES", DEFAULT_GENES),
        n_cells: env_usize("DGE_BENCH_CELLS", DEFAULT_CELLS),
        n_groups: env_usize("DGE_BENCH_GROUPS", DEFAULT_GROUPS),
    };

    let (cell_path, gene_path) = ensure_stores(&shape);
    let cell_reader = ParallelSparseReader::new(cell_path.to_str().unwrap()).expect("cell reader");
    let gene_reader = ParallelSparseReader::new(gene_path.to_str().unwrap()).expect("gene reader");

    let all_cells: Vec<usize> = (0..shape.n_cells).collect();
    let all_genes: Vec<usize> = (0..shape.n_genes).collect();

    let read_cells = || {
        let chunks = cell_reader
            .read_cells_parallel(&all_cells)
            .expect("read cells");
        black_box(chunks.iter().map(|c| c.indices.len()).sum::<usize>())
    };
    let read_genes = || {
        let mut nnz = 0;
        for block in all_genes.chunks(GENE_BLOCK) {
            let chunks = gene_reader.read_gene_parallel(block).expect("read genes");
            nnz += chunks.iter().map(|c| c.indices.len()).sum::<usize>();
        }
        black_box(nnz)
    };

    // warm-up, and the store profile
    let nnz = read_cells();
    assert_eq!(nnz, read_genes(), "cell and gene stores disagree on nnz");
    println!(
        "store: nnz {} ({:.1}% dense, {:.0} nnz per cell)",
        nnz,
        100.0 * nnz as f64 / (shape.n_cells * shape.n_genes) as f64,
        nnz as f64 / shape.n_cells as f64
    );

    let t_cells = best_of(|| {
        read_cells();
    });
    let t_genes = best_of(|| {
        read_genes();
    });

    let grp_1: Vec<usize> = all_cells
        .iter()
        .copied()
        .filter(|&c| group_of(c, shape.n_groups) == 0)
        .collect();
    let grp_2: Vec<usize> = all_cells
        .iter()
        .copied()
        .filter(|&c| group_of(c, shape.n_groups) != 0)
        .collect();

    let t_arm = best_of(|| {
        let res = calculate_dge_grps_mann_whitney(&cell_reader, &grp_1, &grp_2, 0.05, "greater", 0)
            .expect("dge");
        black_box(res.z_scores.len());
    });

    let groups: Vec<Vec<usize>> = (0..shape.n_groups)
        .map(|g| {
            all_cells
                .iter()
                .copied()
                .filter(|&c| group_of(c, shape.n_groups) == g)
                .collect()
        })
        .collect();
    let references: Vec<usize> = (0..shape.n_groups).collect();

    let t_rest = best_of(|| {
        let res = calculate_dge_one_vs_rest_mann_whitney(&gene_reader, &groups, 0.05, "greater", 0)
            .expect("one-vs-rest");
        black_box(res.len());
    });
    let t_many = best_of(|| {
        let res =
            calculate_dge_one_vs_many_auroc(&gene_reader, &groups, &references, 0.05, "greater", 0)
                .expect("one-vs-many");
        black_box(res.len());
    });

    println!("read_cells   {:>10.2?}", t_cells);
    println!("read_genes   {:>10.2?}", t_genes);
    println!(
        "arm_cells    {:>10.2?}  (x{} groups = {:.2?})",
        t_arm,
        shape.n_groups,
        t_arm * shape.n_groups as u32
    );
    println!("scan_rest    {:>10.2?}  (all groups)", t_rest);
    println!("scan_many    {:>10.2?}  (all reference arms)", t_many);
}
