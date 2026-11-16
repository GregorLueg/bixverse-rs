//! Stage split of the ATAC pipeline on a real fragment file.
//!
//! Stages, each timed once:
//!
//! * `scan` - [`fold_fragments`] with a consumer that only counts: inflate
//!   plus parse plus the barcode lookup, the floor of every fragment pass
//! * `qc` - [`fragment_qc`] (needs `ATAC_BENCH_GTF` for the TSSs)
//! * `count` - [`count_fragments`] into the peaks, insertion mode
//! * `write` - [`write_counts_store`]: cell file, gene file, sections
//! * `tfidf` - [`set_tf_idf_on_store`] over all cells
//! * `top` - [`feature_total_stats`] and the q5 cutoff
//! * `lsi` - [`lsi_on_store`], 30 components, `PcaSolver::Auto`
//!
//! Run with:
//! ```text
//! ATAC_BENCH_FRAGMENTS=<fragments.tsv.gz> ATAC_BENCH_PEAKS=<peaks.bed> \
//! ATAC_BENCH_CELLS=<barcodes.txt> ATAC_BENCH_GTF=<genes.gtf.gz> \
//! cargo bench -p bixverse-rs --features atac --bench atac_bench
//! ```
//!
//! `ATAC_BENCH_CELLS` is one barcode per line. `ATAC_BENCH_DIR` sets where the
//! store goes (default: the system temp directory); it is removed at the end.

use std::path::PathBuf;
use std::time::Instant;

use bixverse_rs::single_cell::atac::annotation::{read_gtf_genes, tss_positions};
use bixverse_rs::single_cell::atac::counting::{CountMode, count_fragments, write_counts_store};
use bixverse_rs::single_cell::atac::features::{
    TopFeatureCutoff, feature_total_stats, select_top_features,
};
use bixverse_rs::single_cell::atac::fragments::{BarcodeIndex, fold_fragments};
use bixverse_rs::single_cell::atac::lsi::{
    LsiParams, TF_IDF_SCALE, lsi_on_store, set_tf_idf_on_store,
};
use bixverse_rs::single_cell::atac::qc::{FragmentQcParams, fragment_qc};
use bixverse_rs::single_cell::atac::regions::read_bed;
use bixverse_rs::single_cell::sc_data::data_io::ParallelSparseReader;
use bixverse_rs::single_cell::sc_data::v5_io::FeatureKind;

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

/// Print one stage.
///
/// ### Params
///
/// * `stage` - Stage name
/// * `t` - Start time
/// * `note` - Extra column
fn report(stage: &str, t: Instant, note: &str) {
    println!("| {stage:<6} | {:>9.2} s | {note} |", t.elapsed().as_secs_f64());
}

fn main() {
    let fragments = env_path("ATAC_BENCH_FRAGMENTS");
    let peaks = read_bed(&env_path("ATAC_BENCH_PEAKS")).expect("peaks BED");
    let barcodes: Vec<String> = std::fs::read_to_string(env_path("ATAC_BENCH_CELLS"))
        .expect("barcodes file")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let dir = std::env::var("ATAC_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join(format!("atac_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("bench dir");
    let cf = dir.join("cells.bin").to_str().expect("utf-8").to_string();
    let gf = dir.join("genes.bin").to_str().expect("utf-8").to_string();

    println!(
        "{} cells, {} peaks, {} threads",
        barcodes.len(),
        peaks.len(),
        rayon::current_num_threads()
    );
    println!("| stage  |      time   | note |\n|---|---|---|");

    let t = Instant::now();
    let index = BarcodeIndex::new(&barcodes).expect("barcodes");
    let n_frag = fold_fragments(
        &fragments,
        &index,
        || 0u64,
        |a, _, f| *a += f.len() as u64,
        |a, b| a + b,
    )
    .expect("scan");
    report("scan", t, &format!("{n_frag} fragments of the cells"));

    if let Ok(gtf) = std::env::var("ATAC_BENCH_GTF") {
        let t = Instant::now();
        let genes = read_gtf_genes(&PathBuf::from(gtf)).expect("GTF");
        let tss = tss_positions(&genes, &["protein_coding".to_string()]);
        let gtf_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let qc = fragment_qc(&fragments, &barcodes, &tss, &FragmentQcParams::default())
            .expect("qc");
        let tsse = qc.tss_enrichment();
        let mut sorted = tsse.clone();
        sorted.sort_by(f64::total_cmp);
        report(
            "qc",
            t,
            &format!(
                "{} TSSs (GTF read {gtf_s:.2} s), median TSSe {:.2}",
                tss.len(),
                sorted[sorted.len() / 2]
            ),
        );
    }

    let t = Instant::now();
    let counts =
        count_fragments(&fragments, &barcodes, &peaks, CountMode::Insertion).expect("count");
    let nnz: usize = counts.indices.iter().map(|v| v.len()).sum();
    report("count", t, &format!("{nnz} non-zeros"));

    let t = Instant::now();
    write_counts_store(
        &counts,
        &peaks,
        &barcodes,
        FeatureKind::Peak,
        TF_IDF_SCALE,
        &cf,
        &gf,
        None,
        false,
    )
    .expect("write");
    report("write", t, "");
    drop(counts);

    let t = Instant::now();
    set_tf_idf_on_store(&cf, &gf, None, TF_IDF_SCALE).expect("tfidf");
    report("tfidf", t, "");

    let t = Instant::now();
    let totals = feature_total_stats(&ParallelSparseReader::new(&gf).expect("reader"), None)
        .expect("totals");
    let features = select_top_features(&totals, TopFeatureCutoff::Quantile(0.05));
    report("top", t, &format!("{} features", features.len()));

    let t = Instant::now();
    let cells: Vec<usize> = (0..barcodes.len()).collect();
    let res = lsi_on_store(
        &cf,
        &gf,
        &cells,
        &features,
        &LsiParams {
            n_comp: 30,
            ..Default::default()
        },
        42,
        0,
    )
    .expect("lsi");
    report(
        "lsi",
        t,
        &format!("depth cor of LSI_1 {:.3}", res.depth_cor[0]),
    );

    let _ = std::fs::remove_dir_all(&dir);
}
