//! Gene annotation from a GTF: genes, TSSs and the gene activity regions.
//!
//! Reads the `gene` rows of a GTF (GENCODE or Ensembl, plain or gzip). A gene
//! row spans all of the gene's transcripts, which is what Signac's
//! `CollapseToLongestTranscript` reconstructs from the transcript level, so
//! the gene span is used as is. GTF coordinates are 1-based closed and are
//! converted to BED on reading.

use std::io::BufRead;
use std::path::Path;

use rustc_hash::FxHashMap;

use bixverse_sc_data::store::interval_index::{Arm, FeatureInterval};

use crate::io::open_text;
use crate::prelude::*;

///////////
// Types //
///////////

/// One gene of a GTF.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneRecord {
    /// Gene identifier (`gene_id`).
    pub gene_id: String,
    /// Gene symbol (`gene_name`), the id when absent or empty.
    pub gene_name: String,
    /// Biotype (`gene_type` in GENCODE, `gene_biotype` in Ensembl), empty
    /// when absent.
    pub biotype: String,
    /// Chromosome.
    pub chrom: String,
    /// 0-based start.
    pub start: u32,
    /// Exclusive end.
    pub end: u32,
    /// `1` forward, `-1` reverse, `0` unknown (treated as forward).
    pub strand: i8,
}

impl GeneRecord {
    /// 0-based position of the TSS: the start on the forward strand, the
    /// last base on the reverse strand.
    ///
    /// ### Returns
    ///
    /// The position.
    pub fn tss(&self) -> u32 {
        if self.strand < 0 {
            self.end - 1
        } else {
            self.start
        }
    }
}

/// Parameters of the gene activity regions (Signac `GeneActivity`).
#[derive(Clone, Debug)]
pub struct GeneActivityParams {
    /// Bases added upstream of the gene (strand aware).
    pub upstream: u32,
    /// Bases added downstream of the gene (strand aware).
    pub downstream: u32,
    /// Genes this long or longer are dropped.
    pub max_width: u32,
    /// Biotypes to keep; empty keeps all.
    pub biotypes: Vec<String>,
}

impl Default for GeneActivityParams {
    /// Signac's defaults: 2 kb upstream, nothing downstream, genes under
    /// 500 kb, protein coding only.
    fn default() -> Self {
        Self {
            upstream: 2000,
            downstream: 0,
            max_width: 500_000,
            biotypes: vec!["protein_coding".to_string()],
        }
    }
}

/////////////
// Parsing //
/////////////

/// Value of one attribute in a GTF attribute column.
///
/// ### Params
///
/// * `attrs` - The ninth column
/// * `key` - Attribute name
///
/// ### Returns
///
/// The unquoted value, or `None` when absent.
fn gtf_attribute<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    attrs.split(';').find_map(|kv| {
        let kv = kv.trim();
        let (k, v) = kv.split_once(' ')?;
        (k == key).then(|| v.trim().trim_matches('"'))
    })
}

/// Read the genes of a GTF.
///
/// ### Params
///
/// * `path` - The GTF, plain or gzip (by magic bytes)
///
/// ### Returns
///
/// One [`GeneRecord`] per `gene` row in file order, or an error for a
/// malformed gene row or a file without gene rows.
pub fn read_gtf_genes(path: &Path) -> Result<Vec<GeneRecord>, AtacErrors> {
    let mut genes = Vec::new();
    for line in open_text(path)?.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(9, '\t').collect();
        if f.len() < 9 || f[2] != "gene" {
            continue;
        }
        let bad = || AtacErrors::InvalidArgument(format!("malformed GTF gene row '{line}'"));
        let start: u32 = f[3].parse().map_err(|_| bad())?;
        let end: u32 = f[4].parse().map_err(|_| bad())?;
        if start == 0 || end < start {
            return Err(bad());
        }
        let gene_id = gtf_attribute(f[8], "gene_id").ok_or_else(bad)?.to_string();
        let gene_name = gtf_attribute(f[8], "gene_name")
            .filter(|n| !n.is_empty())
            .unwrap_or(&gene_id)
            .to_string();
        let biotype = gtf_attribute(f[8], "gene_type")
            .or_else(|| gtf_attribute(f[8], "gene_biotype"))
            .unwrap_or("")
            .to_string();
        genes.push(GeneRecord {
            gene_id,
            gene_name,
            biotype,
            chrom: f[0].to_string(),
            start: start - 1,
            end,
            strand: match f[6] {
                "-" => -1,
                "+" => 1,
                _ => 0,
            },
        });
    }
    if genes.is_empty() {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} has no gene rows",
            path.display()
        )));
    }
    Ok(genes)
}

/// Whether a gene passes a biotype filter.
///
/// ### Params
///
/// * `g` - The gene
/// * `biotypes` - Allowed biotypes, empty for all
///
/// ### Returns
///
/// `true` to keep it.
fn keep_biotype(g: &GeneRecord, biotypes: &[String]) -> bool {
    biotypes.is_empty() || biotypes.contains(&g.biotype)
}

/// TSS positions of the genes of some biotypes.
///
/// ### Params
///
/// * `genes` - From [`read_gtf_genes`]
/// * `biotypes` - Biotypes to keep, empty for all (Signac uses protein coding)
///
/// ### Returns
///
/// `(chromosome, 0-based position)` per kept gene, in input order.
pub fn tss_positions(genes: &[GeneRecord], biotypes: &[String]) -> Vec<(String, u32)> {
    genes
        .iter()
        .filter(|g| keep_biotype(g, biotypes))
        .map(|g| (g.chrom.clone(), g.tss()))
        .collect()
}

/// R's `make.unique`: later duplicates get `.1`, `.2`, ...
///
/// ### Params
///
/// * `names` - Names in order
///
/// ### Returns
///
/// Unique names, first occurrences unchanged.
fn make_unique(names: &[String]) -> Vec<String> {
    let mut seen: FxHashMap<String, usize> = FxHashMap::default();
    for n in names {
        seen.entry(n.clone()).or_insert(0);
    }
    let mut counter: FxHashMap<&str, usize> = FxHashMap::default();
    let mut first: FxHashMap<&str, bool> = FxHashMap::default();
    names
        .iter()
        .map(|n| {
            if first.insert(n.as_str(), true).is_none() {
                return n.clone();
            }
            let c = counter.entry(n.as_str()).or_insert(0);
            loop {
                *c += 1;
                let cand = format!("{n}.{c}");
                if !seen.contains_key(&cand) {
                    seen.insert(cand.clone(), 0);
                    return cand;
                }
            }
        })
        .collect()
}

/// Gene body plus promoter regions for gene activity (Signac
/// `GeneActivity`).
///
/// Keeps genes of the requested biotypes shorter than `max_width`, extends
/// them strand aware and clips at the chromosome start.
///
/// ### Params
///
/// * `genes` - From [`read_gtf_genes`]
/// * `params` - See [`GeneActivityParams`]
///
/// ### Returns
///
/// `(regions, gene names, gene ids)`, parallel, names made unique.
pub fn gene_activity_regions(
    genes: &[GeneRecord],
    params: &GeneActivityParams,
) -> (Vec<FeatureInterval>, Vec<String>, Vec<String>) {
    let kept: Vec<&GeneRecord> = genes
        .iter()
        .filter(|g| keep_biotype(g, &params.biotypes) && g.end - g.start < params.max_width)
        .collect();
    let regions = kept
        .iter()
        .map(|g| {
            let (up, down) = (params.upstream, params.downstream);
            let (start, end) = if g.strand < 0 {
                (g.start.saturating_sub(down), g.end.saturating_add(up))
            } else {
                (g.start.saturating_sub(up), g.end.saturating_add(down))
            };
            FeatureInterval {
                chrom: g.chrom.clone(),
                start,
                end,
                strand: if g.strand < 0 { -1 } else { 1 },
                arm: Arm::Unknown,
            }
        })
        .collect();
    let names: Vec<String> = kept.iter().map(|g| g.gene_name.clone()).collect();
    let ids = kept.iter().map(|g| g.gene_id.clone()).collect();
    (regions, make_unique(&names), ids)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A gene for the tests.
    ///
    /// ### Params
    ///
    /// * `name` - Gene name
    /// * `start` - 0-based start
    /// * `end` - Exclusive end
    /// * `strand` - Strand
    /// * `biotype` - Biotype
    ///
    /// ### Returns
    ///
    /// The record.
    fn gene(name: &str, start: u32, end: u32, strand: i8, biotype: &str) -> GeneRecord {
        GeneRecord {
            gene_id: format!("ID_{name}"),
            gene_name: name.to_string(),
            biotype: biotype.to_string(),
            chrom: "chr1".to_string(),
            start,
            end,
            strand,
        }
    }

    #[test]
    fn test_gtf_attribute() {
        let a = r#"gene_id "ENSG1.5"; gene_type "protein_coding"; gene_name "TP53"; level 2;"#;
        assert_eq!(gtf_attribute(a, "gene_id"), Some("ENSG1.5"));
        assert_eq!(gtf_attribute(a, "gene_name"), Some("TP53"));
        assert_eq!(gtf_attribute(a, "level"), Some("2"));
        assert_eq!(gtf_attribute(a, "gene_biotype"), None);
    }

    #[test]
    fn test_tss_is_strand_aware() {
        assert_eq!(gene("A", 100, 200, 1, "").tss(), 100);
        assert_eq!(gene("B", 100, 200, -1, "").tss(), 199);
    }

    #[test]
    fn test_gene_activity_regions_extend_upstream_and_filter() {
        let genes = vec![
            gene("A", 5000, 6000, 1, "protein_coding"),
            gene("B", 5000, 6000, -1, "protein_coding"),
            gene("C", 1000, 2000, 1, "lncRNA"),
            gene("D", 500, 600_000, 1, "protein_coding"),
            gene("A", 100, 200, 1, "protein_coding"),
        ];
        let (regions, names, ids) = gene_activity_regions(&genes, &GeneActivityParams::default());
        assert_eq!(ids, vec!["ID_A", "ID_B", "ID_A"]);
        let got: Vec<(u32, u32)> = regions.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(got, vec![(3000, 6000), (5000, 8000), (0, 200)]);
        assert_eq!(names, vec!["A", "B", "A.1"]);
    }

    #[test]
    fn test_make_unique_matches_r() {
        // R: make.unique(c("a", "a", "b", "a", "a.1"))
        let names: Vec<String> = ["a", "a", "b", "a", "a.1"].map(String::from).to_vec();
        assert_eq!(make_unique(&names), vec!["a", "a.2", "b", "a.3", "a.1"]);
    }
}
