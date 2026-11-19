//! Peak calling on a real fragment file, timed against the scan floor.
//!
//! Stages, each timed once:
//!
//! * `scan` - [`fold_fragments`] with a consumer that only counts: inflate
//!   plus parse plus the barcode lookup, the floor of every fragment pass
//! * `call` - [`call_peaks`] end to end: tag pass, pileups, scores, peaks
//!
//! Run with:
//! ```text
//! PEAK_BENCH_FRAGMENTS=<fragments.tsv.gz> PEAK_BENCH_CELLS=<barcodes.txt> \
//! cargo bench -p bixverse-rs --features atac --bench peak_call_bench
//! ```
//!
//! Optional: `PEAK_BENCH_GROUPS` (one group index per barcode line, default
//! all in group 0), `PEAK_BENCH_TAG` (`insertion`, the default, or
//! `fragment_start`), `PEAK_BENCH_KEEP_DUP` (`all`, the default, or a count),
//! `PEAK_BENCH_MAX_MEM_GB` (tag budget, default one chromosome at a time),
//! `PEAK_BENCH_SKIP_SCAN` (set to skip the scan, so the peak RSS is the call's),
//! `PEAK_BENCH_OUT` (write the peaks as TSV: group, chrom, start, end,
//! summit, pileup, pscore, fold, qscore).

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use bixverse_rs::single_cell::atac::io::fragments::{BarcodeIndex, fold_fragments};
use bixverse_rs::single_cell::atac::peaks::calling::{
    KeepDup, PeakCallParams, call_peaks, parse_tag_source,
};

/// A required environment variable as a path.
///
/// ### Params
///
/// * `name` - Variable name
///
/// ### Returns
///
/// The path; panics with the variable's name when unset.
fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

/// Non-empty trimmed lines of a file.
///
/// ### Params
///
/// * `path` - The file
///
/// ### Returns
///
/// The lines.
fn read_lines(path: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn main() {
    let fragments = env_path("PEAK_BENCH_FRAGMENTS");
    let barcodes = read_lines(&env_path("PEAK_BENCH_CELLS"));
    let groups: Vec<u32> = match std::env::var("PEAK_BENCH_GROUPS") {
        Ok(p) => read_lines(&PathBuf::from(p))
            .iter()
            .map(|g| g.parse().expect("integer group"))
            .collect(),
        Err(_) => vec![0; barcodes.len()],
    };
    let tag_source =
        parse_tag_source(&std::env::var("PEAK_BENCH_TAG").unwrap_or_else(|_| "insertion".into()))
            .expect("PEAK_BENCH_TAG");
    let keep_dup = match std::env::var("PEAK_BENCH_KEEP_DUP").as_deref() {
        Err(_) | Ok("all") => KeepDup::All,
        Ok(n) => KeepDup::AtMost(n.parse().expect("integer keep_dup")),
    };
    println!(
        "{} cells, {} groups, {tag_source:?}, {keep_dup:?}, {} threads",
        barcodes.len(),
        groups.iter().max().map_or(0, |&g| g + 1),
        rayon::current_num_threads()
    );
    println!("| stage  | wall        | note |");
    println!("|--------|-------------|------|");

    // the scan maps the whole file, which shows in the peak RSS of the call
    if std::env::var("PEAK_BENCH_SKIP_SCAN").is_err() {
        let index = BarcodeIndex::new(&barcodes).expect("barcodes");
        let t = Instant::now();
        let n: usize = fold_fragments(
            &fragments,
            &index,
            || 0usize,
            |acc, _, frags| *acc += frags.len(),
            |a, b| a + b,
        )
        .expect("scan");
        println!(
            "| scan   | {:>9.2} s | {n} fragments |",
            t.elapsed().as_secs_f64()
        );
    }

    let params = PeakCallParams {
        tag_source,
        keep_dup,
        max_mem_gb: std::env::var("PEAK_BENCH_MAX_MEM_GB")
            .ok()
            .map(|v| v.parse().expect("PEAK_BENCH_MAX_MEM_GB")),
        ..PeakCallParams::default()
    };
    let t = Instant::now();
    let peaks = call_peaks(&fragments, &barcodes, &groups, Some(params)).expect("call");
    let n_peaks: usize = peaks.iter().map(|p| p.len()).sum();
    println!(
        "| call   | {:>9.2} s | {n_peaks} peaks |",
        t.elapsed().as_secs_f64()
    );

    if let Ok(out) = std::env::var("PEAK_BENCH_OUT") {
        let mut w = std::io::BufWriter::new(std::fs::File::create(out).expect("out file"));
        for (g, ps) in peaks.iter().enumerate() {
            for p in ps {
                writeln!(
                    w,
                    "{g}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    p.chrom, p.start, p.end, p.summit, p.pileup, p.pscore, p.fold, p.qscore
                )
                .expect("write");
            }
        }
    }
}
