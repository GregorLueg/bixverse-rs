//! Parity with Signac, Cell Ranger and SnapATAC2 on 500 PBMC multiome cells,
//! chr22 only. Fixtures and their generators: `tests/fixtures/pbmc3k_chr22/`
//! and `dev/atac/`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;

use bixverse_atac::annotation::{GeneActivityParams, gene_activity_regions, read_gtf_genes};
use bixverse_atac::counting::{CountMode, FeatureCounts, count_fragments, write_counts_store};
use bixverse_atac::features::{TopFeatureCutoff, feature_total_stats, select_top_features};
use bixverse_atac::lsi::{LsiParams, TF_IDF_SCALE, lsi_on_store, set_tf_idf_on_store};
use bixverse_atac::qc::{FragmentQcParams, fragment_qc};
use bixverse_atac::regions::{genome_tiles, read_bed, region};
use bixverse_sc::sc_processing::pca::PcaSolver;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_data::interval_index::FeatureInterval;
use bixverse_sc_data::v5_io::FeatureKind;

/// chr22 length in hg38, as in `chrom_sizes.tsv`.
const CHR22_LEN: u32 = 50_818_468;

/////////////
// Helpers //
/////////////

/// The fixture directory.
///
/// ### Returns
///
/// Its path.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pbmc3k_chr22")
}

/// Lines of a plain or gzip text file, header included.
///
/// ### Params
///
/// * `rel` - Path relative to the fixture directory
///
/// ### Returns
///
/// The lines.
fn lines(rel: &str) -> Vec<String> {
    let path = fixtures().join(rel);
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let reader: Box<dyn BufRead> = if rel.ends_with(".gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    reader.lines().map(|l| l.unwrap()).collect()
}

/// Rows of a TSV with a header, split into fields.
///
/// ### Params
///
/// * `rel` - Path relative to the fixture directory
///
/// ### Returns
///
/// `(header, rows)`.
fn tsv(rel: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let all = lines(rel);
    let header = all[0].split('\t').map(String::from).collect();
    let rows = all[1..]
        .iter()
        .map(|l| l.split('\t').map(String::from).collect())
        .collect();
    (header, rows)
}

/// One column of a TSV as `f64` (`NaN` parses).
///
/// ### Params
///
/// * `rel` - Path relative to the fixture directory
/// * `name` - Column name
///
/// ### Returns
///
/// The values in row order.
fn column(rel: &str, name: &str) -> Vec<f64> {
    let (header, rows) = tsv(rel);
    let j = header
        .iter()
        .position(|h| h == name)
        .expect("column exists");
    rows.iter().map(|r| r[j].parse().expect("number")).collect()
}

/// The 500 barcodes.
///
/// ### Returns
///
/// The barcodes in fixture order.
fn cells() -> Vec<String> {
    lines("cells.txt")
}

/// The fragment file.
///
/// ### Returns
///
/// Its path.
fn fragments() -> PathBuf {
    fixtures().join("fragments.tsv.gz")
}

/// A sparse fixture matrix as sorted `(feature, cell, count)` triplets.
///
/// ### Params
///
/// * `rel` - Path of the `matrix.tsv.gz`
///
/// ### Returns
///
/// The triplets.
fn triplets(rel: &str) -> Vec<(u32, u32, u32)> {
    let (_, rows) = tsv(rel);
    let mut v: Vec<(u32, u32, u32)> = rows
        .iter()
        .map(|r| {
            (
                r[0].parse().unwrap(),
                r[1].parse().unwrap(),
                r[2].parse().unwrap(),
            )
        })
        .collect();
    v.sort_unstable();
    v
}

/// Our counts as sorted `(feature, cell, count)` triplets.
///
/// ### Params
///
/// * `c` - The counts
///
/// ### Returns
///
/// The triplets.
fn our_triplets(c: &FeatureCounts) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = c
        .indices
        .iter()
        .zip(&c.counts)
        .enumerate()
        .flat_map(|(cell, (idx, cnt))| idx.iter().zip(cnt).map(move |(&f, &n)| (f, cell as u32, n)))
        .collect();
    v.sort_unstable();
    v
}

/// Features of a `features.tsv.gz` (`chrom start end name`).
///
/// ### Params
///
/// * `rel` - Path relative to the fixture directory
///
/// ### Returns
///
/// `(intervals, names)`.
fn feature_file(rel: &str) -> (Vec<FeatureInterval>, Vec<String>) {
    let (_, rows) = tsv(rel);
    let iv = rows
        .iter()
        .map(|r| region(&r[0], r[1].parse().unwrap(), r[2].parse().unwrap()))
        .collect();
    let names = rows.iter().map(|r| r[3].clone()).collect();
    (iv, names)
}

/// Scratch directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// A fresh directory.
    ///
    /// ### Params
    ///
    /// * `tag` - Unique name part
    ///
    /// ### Returns
    ///
    /// The guard.
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("bv_atac_it_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    /// A file path inside, as `String`.
    ///
    /// ### Params
    ///
    /// * `name` - File name
    ///
    /// ### Returns
    ///
    /// The path.
    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Assert two values agree, treating two `NaN`s as equal.
///
/// ### Params
///
/// * `ours` - Our value
/// * `theirs` - Reference
/// * `tol` - Absolute tolerance
/// * `what` - Context
fn close(ours: f64, theirs: f64, tol: f64, what: &str) {
    if theirs.is_nan() {
        assert!(ours.is_nan(), "{what}: {ours} vs NaN");
    } else {
        assert!((ours - theirs).abs() <= tol, "{what}: {ours} vs {theirs}");
    }
}

//////////////
// Counting //
//////////////

#[test]
fn test_fragment_counts_match_signac_feature_matrix() {
    let peaks = read_bed(&fixtures().join("peaks.bed")).unwrap();
    let c = count_fragments(&fragments(), &cells(), &peaks, CountMode::Fragment).unwrap();
    assert_eq!(our_triplets(&c), triplets("signac/peaks/matrix.tsv.gz"));
}

#[test]
fn test_insertion_counts_match_cell_ranger_peaks() {
    let (peaks, _) = feature_file("tenx/features.tsv.gz");
    let c = count_fragments(&fragments(), &cells(), &peaks, CountMode::Insertion).unwrap();
    assert_eq!(our_triplets(&c), triplets("tenx/matrix.tsv.gz"));
}

#[test]
fn test_fragment_counts_match_signac_genome_bins() {
    let tiles = genome_tiles(&[("chr22".to_string(), CHR22_LEN)], 5000);
    let c = count_fragments(&fragments(), &cells(), &tiles, CountMode::Fragment).unwrap();
    assert_eq!(our_triplets(&c), triplets("signac/bins_5000/matrix.tsv.gz"));
}

#[test]
fn test_fragment_counts_match_snapatac2_tiles() {
    let tiles = genome_tiles(&[("chr22".to_string(), CHR22_LEN)], 500);
    let c = count_fragments(&fragments(), &cells(), &tiles, CountMode::Fragment).unwrap();
    assert_eq!(
        our_triplets(&c),
        triplets("snapatac2/tiles_500_fragment/matrix.tsv.gz")
    );
}

////////
// QC //
////////

#[test]
fn test_fragment_qc_matches_signac() {
    let tss: Vec<(String, u32)> = lines("signac/tss.bed")
        .iter()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), f[1].parse().unwrap())
        })
        .collect();
    let qc = fragment_qc(&fragments(), &cells(), &tss, &FragmentQcParams::default()).unwrap();
    let m = "signac/cell_metrics.tsv";

    let as_f64 = |v: &[u64]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    assert_eq!(as_f64(&qc.n_fragments), column(m, "frequency_count"));
    assert_eq!(as_f64(&qc.n_reads), column(m, "reads_count"));
    assert_eq!(as_f64(&qc.nucleosome_free), column(m, "nucleosome_free"));
    assert_eq!(as_f64(&qc.mononucleosomal), column(m, "mononucleosomal"));
    assert_eq!(
        as_f64(&qc.n_fragments),
        column("snapatac2/cell_metrics.tsv", "n_fragment")
            .iter()
            .map(|x| if x.is_nan() { 0.0 } else { *x })
            .collect::<Vec<_>>()
    );

    for (i, (ours, theirs)) in qc
        .nucleosome_signal()
        .iter()
        .zip(column(m, "nucleosome_signal"))
        .enumerate()
    {
        close(
            *ours,
            theirs,
            1e-12,
            &format!("nucleosome signal, cell {i}"),
        );
    }
    for (i, (ours, theirs)) in qc
        .tss_enrichment()
        .iter()
        .zip(column(m, "tss_enrichment_fast"))
        .enumerate()
    {
        close(*ours, theirs, 1e-9, &format!("TSS enrichment, cell {i}"));
    }
}

////////////////////////
// Features and LSI   //
////////////////////////

/// The Signac peak matrix as a TF-IDF store.
///
/// ### Params
///
/// * `dir` - Where the store goes
///
/// ### Returns
///
/// `(cell file, gene file)`.
fn signac_peak_store(dir: &TempDir) -> (String, String) {
    let peaks = read_bed(&fixtures().join("peaks.bed")).unwrap();
    let barcodes = cells();
    let c = count_fragments(&fragments(), &barcodes, &peaks, CountMode::Fragment).unwrap();
    let (cf, gf) = (dir.file("cells.bin"), dir.file("genes.bin"));
    write_counts_store(
        &c,
        &peaks,
        &barcodes,
        FeatureKind::Peak,
        TF_IDF_SCALE,
        &cf,
        &gf,
        None,
        false,
    )
    .unwrap();
    (cf, gf)
}

#[test]
fn test_top_features_match_signac() {
    let dir = TempDir::new("top");
    let (_, gf) = signac_peak_store(&dir);
    let totals = feature_total_stats(&ParallelSparseReader::new(&gf).unwrap(), None).unwrap();
    let f = "signac/top_features.tsv";
    assert_eq!(totals.count, column(f, "count"));
    for (ours, theirs) in totals.percentile.iter().zip(column(f, "percentile")) {
        close(*ours, theirs, 1e-12, "percentile");
    }
    let kept: Vec<usize> = column(f, "kept")
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == 1.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        select_top_features(&totals, TopFeatureCutoff::Quantile(0.05)),
        kept
    );
}

#[test]
fn test_lsi_matches_signac_run_svd() {
    let dir = TempDir::new("lsi");
    let (cf, gf) = signac_peak_store(&dir);
    let n_cells = cells().len();
    set_tf_idf_on_store(&cf, &gf, None, TF_IDF_SCALE).unwrap();
    let totals = feature_total_stats(&ParallelSparseReader::new(&gf).unwrap(), None).unwrap();
    let features = select_top_features(&totals, TopFeatureCutoff::Quantile(0.05));

    let all: Vec<usize> = (0..n_cells).collect();
    let params = LsiParams::new(10, PcaSolver::Exact, true, None);
    let res = lsi_on_store(&cf, &gf, &all, &features, &params, 42, 0).unwrap();

    let comp = "signac/lsi_components.tsv";
    for (j, (ours, theirs)) in res
        .singular_values
        .iter()
        .zip(column(comp, "d"))
        .enumerate()
    {
        let rel = (*ours as f64 - theirs).abs() / theirs;
        assert!(rel < 1e-4, "d[{j}]: {ours} vs {theirs}");
    }

    let (_, rows) = tsv("signac/lsi_embeddings.tsv.gz");
    for j in 0..10 {
        let theirs: Vec<f64> = rows.iter().map(|r| r[j + 1].parse().unwrap()).collect();
        let ours: Vec<f64> = (0..n_cells).map(|i| res.embedding[(i, j)] as f64).collect();
        let sign = if ours.iter().zip(&theirs).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
            -1.0
        } else {
            1.0
        };
        let max_diff = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a * sign - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-3, "LSI_{}: max |diff| {max_diff}", j + 1);

        let theirs_cor = column(comp, "depth_cor")[j];
        close(
            res.depth_cor[j] as f64 * sign,
            theirs_cor,
            1e-4,
            &format!("depth cor {}", j + 1),
        );
    }
}

///////////////////
// Gene activity //
///////////////////

#[test]
fn test_gene_activity_regions_match_signac() {
    let genes = read_gtf_genes(&fixtures().join("genes.gtf.gz")).unwrap();
    let (regions, names) = gene_activity_regions(&genes, &GeneActivityParams::default());
    let mut ours: Vec<(String, String, u32, u32)> = regions
        .iter()
        .zip(&names)
        .map(|(r, n)| (n.clone(), r.chrom.clone(), r.start, r.end))
        .collect();
    let (theirs_iv, theirs_names) = feature_file("signac/gene_activity/features.tsv.gz");
    let mut theirs: Vec<(String, String, u32, u32)> = theirs_iv
        .iter()
        .zip(&theirs_names)
        .map(|(r, n)| (n.clone(), r.chrom.clone(), r.start, r.end))
        .collect();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs);
}

#[test]
fn test_gene_activity_counts_match_signac() {
    let (regions, _) = feature_file("signac/gene_activity/features.tsv.gz");
    let c = count_fragments(&fragments(), &cells(), &regions, CountMode::Fragment).unwrap();
    assert_eq!(
        our_triplets(&c),
        triplets("signac/gene_activity/matrix.tsv.gz")
    );
}
