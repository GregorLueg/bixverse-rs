//! Leiden and Louvain on a real sNN graph: time against iteration count, and
//! what the extra iterations buy.
//!
//! Input is a `SparseGraph` blob as R's `to_bytes()` writes it (f32 or f64
//! weights). Stages, each timed over interleaved repeats (median reported):
//!
//! * `level0` - one iteration with `multi_level: false`: the BFS relabelling
//!   plus the first level-0 local moving, nothing else.
//! * `iters` - Leiden with `n_iterations` 1 to `LEIDEN_BENCH_MAX_ITERS`. Per
//!   count: time, marginal time, modularity, cluster count, ARI against the
//!   largest count and against the truth.
//! * `stable` - Leiden until stable, with `min_gain` 0 and 1e-4.
//! * `louvain` - Louvain at R's default of 2 iterations.
//!
//! Every row prints an `FxHash` of its labels, so two builds of the engine can
//! be checked for identical partitions.
//!
//! Run with:
//! ```text
//! LEIDEN_BENCH_GRAPH=/path/graph.bin cargo bench -p bixverse-rs --bench leiden_bench
//! ```
//!
//! Environment:
//!
//! * `LEIDEN_BENCH_GRAPH` (required) - the graph blob
//! * `LEIDEN_BENCH_TRUTH` - one integer label per node, per line
//! * `LEIDEN_BENCH_REPS` - timed repeats per configuration, default 3
//! * `LEIDEN_BENCH_MAX_ITERS` - largest iteration count, default 5
//! * `LEIDEN_BENCH_RES` - resolution, default 1
//! * `LEIDEN_BENCH_SEED` - seed, default 42
//! * `LEIDEN_BENCH_ONLY` - comma-separated subset of
//!   `level0,iters,stable,louvain`

use std::hash::BuildHasher;
use std::hint::black_box;
use std::time::Instant;

use bixverse_rs::graph::graph_structures::SparseGraph;
use bixverse_rs::graph::leiden::{LeidenParams, community_detection, modularity};
use bixverse_rs::ml::clustering::clustering_metrics::adjusted_rand_index;
use bixverse_rs::prelude::*;
use rustc_hash::FxBuildHasher;

/// One configuration of the engine.
struct Config {
    /// Row label
    name: String,
    /// Engine parameters
    params: LeidenParams,
    /// Stop after the first local-moving phase
    multi_level: bool,
}

/// Wall seconds, run in a fixed order.
///
/// ### Params
///
/// * `g` - The graph
/// * `c` - The configuration
///
/// ### Returns
///
/// `(labels, seconds)`
fn run<T>(g: &SparseGraph<T>, c: &Config) -> (Vec<usize>, f64)
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let t = Instant::now();
    let labels = community_detection(black_box(g), &c.params, usize::MAX, c.multi_level)
        .expect("community_detection");
    (black_box(labels), t.elapsed().as_secs_f64())
}

/// Median of a slice.
///
/// ### Params
///
/// * `v` - Values
///
/// ### Returns
///
/// The median
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

/// Number of distinct labels.
///
/// ### Params
///
/// * `l` - Labels
///
/// ### Returns
///
/// The count
fn n_clusters(l: &[usize]) -> usize {
    l.iter().max().map_or(0, |m| m + 1)
}

/// Print the load average next to the timings.
fn uptime() {
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprint!("{}", String::from_utf8_lossy(&o.stdout));
    }
}

/// Run every selected stage on one graph.
///
/// ### Params
///
/// * `g` - The graph
/// * `truth` - Truth label per node, if given
fn bench<T>(g: &SparseGraph<T>, truth: Option<&[usize]>)
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let reps = env("LEIDEN_BENCH_REPS", 3).max(1);
    let max_iters = env("LEIDEN_BENCH_MAX_ITERS", 5).max(1);
    let res: f64 = std::env::var("LEIDEN_BENCH_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let only = std::env::var("LEIDEN_BENCH_ONLY").unwrap_or_default();
    let want = |s: &str| only.is_empty() || only.split(',').any(|o| o.trim() == s);

    let seed = env("LEIDEN_BENCH_SEED", 42) as u64;
    let leiden = |iters: Option<usize>, min_gain: f64| LeidenParams {
        resolution: res,
        seed,
        n_iterations: iters,
        min_gain,
        ..LeidenParams::default()
    };

    let mut configs = Vec::new();
    if want("level0") {
        configs.push(Config {
            name: "level0 only".into(),
            params: leiden(Some(1), 0.0),
            multi_level: false,
        });
    }
    if want("iters") {
        for k in 1..=max_iters {
            configs.push(Config {
                name: format!("leiden it={k}"),
                params: leiden(Some(k), 0.0),
                multi_level: true,
            });
        }
    }
    if want("stable") {
        for mg in [0.0, 1e-4] {
            configs.push(Config {
                name: format!("leiden stable gain={mg:e}"),
                params: leiden(None, mg),
                multi_level: true,
            });
        }
    }
    if want("louvain") {
        configs.push(Config {
            name: "louvain it=2".into(),
            params: LeidenParams {
                refine: false,
                ..leiden(Some(2), 0.0)
            },
            multi_level: true,
        });
    }

    // interleaved: every configuration once per round
    let mut times = vec![Vec::with_capacity(reps); configs.len()];
    let mut labels: Vec<Vec<usize>> = vec![Vec::new(); configs.len()];
    uptime();
    for rep in 0..reps {
        for (i, c) in configs.iter().enumerate() {
            let (l, t) = run(g, c);
            times[i].push(t);
            if rep == 0 {
                labels[i] = l;
            } else {
                assert_eq!(labels[i], l, "{}: labels differ between repeats", c.name);
            }
        }
    }
    uptime();

    let reference = configs
        .iter()
        .position(|c| c.name == format!("leiden it={max_iters}"));
    println!(
        "{:<26} {:>9} {:>9} {:>10} {:>6} {:>9} {:>9} {:>17}",
        "config", "median s", "marg s", "modularity", "k", "ARI ref", "ARI truth", "labels hash"
    );
    let mut prev_iter_time = 0.0;
    for (i, c) in configs.iter().enumerate() {
        let t = median(&times[i]);
        let marg = if c.name.starts_with("leiden it=") {
            let m = t - prev_iter_time;
            prev_iter_time = t;
            format!("{m:.3}")
        } else {
            "-".into()
        };
        let q = modularity(g, &labels[i], res);
        let ari_ref = reference.map_or("-".into(), |r| {
            format!("{:.4}", adjusted_rand_index(&labels[r], &labels[i]))
        });
        let ari_truth = truth.map_or("-".into(), |tr| {
            format!("{:.4}", adjusted_rand_index(tr, &labels[i]))
        });
        println!(
            "{:<26} {:>9.3} {:>9} {:>10.6} {:>6} {:>9} {:>9} {:>17x}",
            c.name,
            t,
            marg,
            q,
            n_clusters(&labels[i]),
            ari_ref,
            ari_truth,
            FxBuildHasher.hash_one(&labels[i])
        );
    }
}

/// Describe and bench the graph in its stored width.
///
/// ### Params
///
/// * `g` - The graph
/// * `truth` - Truth labels
fn describe_and_bench<T>(g: &SparseGraph<T>, truth: Option<&[usize]>)
where
    T: BixverseFloat + Send + Sync + std::iter::Sum,
{
    let n = g.get_node_number();
    let degs: Vec<usize> = (0..n).map(|i| g.get_neighbours_raw(i).0.len()).collect();
    let nnz: usize = degs.iter().sum();
    eprintln!(
        "graph: {n} nodes, {nnz} directed entries, mean degree {:.1}, max degree {}, {} threads",
        nnz as f64 / n.max(1) as f64,
        degs.iter().max().unwrap_or(&0),
        std::thread::available_parallelism().map_or(0, |p| p.get())
    );
    if let Some(tr) = truth {
        assert_eq!(tr.len(), n, "truth has {} labels for {n} nodes", tr.len());
    }
    bench(g, truth);
}

fn main() {
    let path = std::env::var("LEIDEN_BENCH_GRAPH").expect("set LEIDEN_BENCH_GRAPH");
    let bytes = std::fs::read(&path).expect("read graph");
    let truth: Option<Vec<usize>> = std::env::var("LEIDEN_BENCH_TRUTH").ok().map(|p| {
        std::fs::read_to_string(p)
            .expect("read truth")
            .lines()
            .map(|l| l.trim().parse().expect("integer label"))
            .collect()
    });

    if let Ok(g) = SparseGraph::<f32>::from_bytes(&bytes) {
        describe_and_bench(&g, truth.as_deref());
    } else if let Ok(g) = SparseGraph::<f64>::from_bytes(&bytes) {
        describe_and_bench(&g, truth.as_deref());
    } else {
        panic!("{path}: not a SparseGraph blob of f32 or f64 weights");
    }
}
