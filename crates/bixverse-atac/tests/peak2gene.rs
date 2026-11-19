//! Peak to gene links against numpy and scipy (`dev/atac/gen_peak2gene.py`).

use std::path::PathBuf;

use faer::Mat;

use bixverse_atac::peaks::regions::region;
use bixverse_atac::processing::links::{LinkParams, peak_to_gene_links};

/// Lines of a fixture file, header dropped.
fn rows(name: &str) -> Vec<Vec<String>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/peak2gene")
        .join(name);
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(|l| l.split('\t').map(String::from).collect())
        .collect()
}

/// A tab-separated integer matrix as `f64`.
fn matrix(name: &str) -> Mat<f64> {
    let r = rows(name);
    Mat::from_fn(r.len(), r[0].len(), |i, j| r[i][j].parse().unwrap())
}

#[test]
fn test_peak_to_gene_links_match_scipy() {
    let peaks: Vec<_> = rows("peaks.tsv")[1..]
        .iter()
        .map(|r| region(&r[0], r[1].parse().unwrap(), r[2].parse().unwrap()))
        .collect();
    let tss: Vec<(String, u32)> = rows("tss.tsv")[1..]
        .iter()
        .map(|r| (r[0].clone(), r[1].parse().unwrap()))
        .collect();
    let params = LinkParams {
        max_distance: 50_000,
        ..LinkParams::default()
    };
    let ours = peak_to_gene_links(
        matrix("peak_counts.tsv").as_ref(),
        matrix("gene_counts.tsv").as_ref(),
        &peaks,
        &tss,
        &params,
    )
    .unwrap();
    let theirs = &rows("links.tsv")[1..];
    assert_eq!(ours.len(), theirs.len());
    for (o, t) in ours.iter().zip(theirs) {
        assert_eq!(o.peak, t[0].parse::<u32>().unwrap());
        assert_eq!(o.gene, t[1].parse::<u32>().unwrap());
        assert_eq!(o.distance, t[2].parse::<i64>().unwrap());
        let close = |a: f64, b: f64, what: &str| {
            assert!(
                (a - b).abs() <= 1e-10 * b.abs().max(1e-12),
                "peak {} gene {} {what}: {a} vs {b}",
                o.peak,
                o.gene
            )
        };
        close(o.r, t[3].parse().unwrap(), "r");
        close(o.p, t[4].parse().unwrap(), "p");
        close(o.fdr, t[5].parse().unwrap(), "fdr");
    }
    // the planted links are found
    assert!(
        ours.iter()
            .any(|l| l.peak == 0 && l.gene == 0 && l.fdr < 0.05)
    );
    assert!(
        ours.iter()
            .any(|l| l.peak == 5 && l.gene == 1 && l.fdr < 0.05)
    );
}
