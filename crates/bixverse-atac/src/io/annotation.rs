//! Gene annotation from a GTF: genes, TSSs and the gene activity regions.
//!
//! Reads the `gene` rows of a GTF (GENCODE or Ensembl, plain or gzip). A gene
//! row spans all of the gene's transcripts, which is what Signac's
//! `CollapseToLongestTranscript` reconstructs from the transcript level, so
//! the gene span is used as is. GTF coordinates are 1-based closed and are
//! converted to BED on reading.

use std::io::BufRead;
use std::path::Path;

use rayon::prelude::*;
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
    ///
    /// ### Returns
    ///
    /// The defaults.
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
/// GENCODE and Ensembl GTFs carry `gene` rows. UCSC and RefSeq style GTFs
/// often carry only `transcript` (or only `exon`) rows; those are collapsed
/// per `gene_id` to the span of all their rows, as Signac does with
/// transcripts. The biotype is `gene_type` or `gene_biotype`, else
/// `transcript_type` or `transcript_biotype`, else empty.
///
/// ### Params
///
/// * `path` - The GTF, plain or gzip (by magic bytes)
///
/// ### Returns
///
/// One [`GeneRecord`] per gene in file order of first appearance, or an
/// error for a malformed row or a file with none of the three row types.
pub fn read_gtf_genes(path: &Path) -> Result<Vec<GeneRecord>, AtacErrors> {
    for feature in ["gene", "transcript", "exon"] {
        let genes = read_gtf_rows(path, feature)?;
        if !genes.is_empty() {
            return Ok(genes);
        }
    }
    Err(AtacErrors::InvalidArgument(format!(
        "{} has no gene, transcript or exon rows",
        path.display()
    )))
}

/// The rows of one feature type of a GTF, one [`GeneRecord`] per `gene_id`
/// spanning all its rows.
///
/// ### Params
///
/// * `path` - The GTF
/// * `feature` - Third-column value to read
///
/// ### Returns
///
/// The genes in order of first appearance, or an error for a malformed row.
fn read_gtf_rows(path: &Path, feature: &str) -> Result<Vec<GeneRecord>, AtacErrors> {
    let mut genes: Vec<GeneRecord> = Vec::new();
    let mut by_id: FxHashMap<String, usize> = FxHashMap::default();
    for line in open_text(path)?.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(9, '\t').collect();
        if f.len() < 9 || f[2] != feature {
            continue;
        }
        let bad = || AtacErrors::InvalidArgument(format!("malformed GTF {feature} row '{line}'"));
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
        let biotype = [
            "gene_type",
            "gene_biotype",
            "transcript_type",
            "transcript_biotype",
        ]
        .iter()
        .find_map(|k| gtf_attribute(f[8], k))
        .unwrap_or("")
        .to_string();
        // gene rows stand as they are; transcript and exon rows of one gene on
        // one chromosome collapse to their span
        if let Some(&i) = by_id
            .get(&gene_id)
            .filter(|&&i| feature != "gene" && genes[i].chrom == f[0])
        {
            let g = &mut genes[i];
            g.start = g.start.min(start - 1);
            g.end = g.end.max(end);
            continue;
        }
        by_id.insert(gene_id.clone(), genes.len());
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

/// Error when a biotype filter keeps none of the genes.
///
/// A GTF without biotypes (RefSeq) would otherwise turn the default
/// protein-coding filter into an empty TSS set and all-zero scores.
///
/// ### Params
///
/// * `genes` - The genes
/// * `biotypes` - The filter
///
/// ### Returns
///
/// `Ok(())`, or [`AtacErrors::InvalidArgument`] naming the biotypes present.
fn check_biotypes(genes: &[GeneRecord], biotypes: &[String]) -> Result<(), AtacErrors> {
    if genes.is_empty() || genes.iter().any(|g| keep_biotype(g, biotypes)) {
        return Ok(());
    }
    let mut present: Vec<&str> = genes.iter().map(|g| g.biotype.as_str()).collect();
    present.sort_unstable();
    present.dedup();
    let present = if present == [""] {
        "none (the GTF carries no biotype attributes); pass an empty biotype list".to_string()
    } else {
        present
            .iter()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(AtacErrors::InvalidArgument(format!(
        "no gene has a biotype in {biotypes:?}; the GTF has: {present}"
    )))
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
/// `(chromosome, 0-based position)` per kept gene, in input order, or an
/// error when the filter keeps no gene.
pub fn tss_positions(
    genes: &[GeneRecord],
    biotypes: &[String],
) -> Result<Vec<(String, u32)>, AtacErrors> {
    check_biotypes(genes, biotypes)?;
    Ok(genes
        .iter()
        .filter(|g| keep_biotype(g, biotypes))
        .map(|g| (g.chrom.clone(), g.tss()))
        .collect())
}

/// TSSs with their strand, for strand-aware profiles.
///
/// ### Params
///
/// * `genes` - From [`read_gtf_genes`]
/// * `biotypes` - Biotypes to keep, empty for all
///
/// ### Returns
///
/// `(chromosome, 0-based position, strand)` per kept gene, in input order,
/// or an error when the filter keeps no gene.
pub fn tss_sites(
    genes: &[GeneRecord],
    biotypes: &[String],
) -> Result<Vec<(String, u32, i8)>, AtacErrors> {
    check_biotypes(genes, biotypes)?;
    Ok(genes
        .iter()
        .filter(|g| keep_biotype(g, biotypes))
        .map(|g| (g.chrom.clone(), g.tss(), g.strand))
        .collect())
}

/// The gene closest to a region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosestGene {
    /// Index of the gene in the gene list, `None` on a chromosome without
    /// genes.
    pub gene: Option<usize>,
    /// Bases between the region and the gene body, 0 when they overlap or
    /// touch.
    pub distance: u32,
    /// Signed distance from the region's centre to the gene's TSS, positive
    /// downstream of the TSS on the gene's strand.
    pub tss_distance: i64,
}

/// The closest gene body to every region (Signac `ClosestFeature`, on gene
/// bodies).
///
/// Overlapping genes are at distance 0; among several, the first in the
/// gene list wins, as among equally distant ones.
///
/// ### Params
///
/// * `regions` - The regions, e.g. peaks
/// * `genes` - From [`read_gtf_genes`], already filtered as wanted
///
/// ### Returns
///
/// One [`ClosestGene`] per region, in region order.
pub fn closest_genes(regions: &[FeatureInterval], genes: &[GeneRecord]) -> Vec<ClosestGene> {
    // per chromosome: genes sorted by start, by end, and the longest gene
    let mut by_chrom: FxHashMap<&str, (Vec<usize>, Vec<usize>, u32)> = FxHashMap::default();
    for (i, g) in genes.iter().enumerate() {
        let e = by_chrom.entry(g.chrom.as_str()).or_default();
        e.0.push(i);
        e.1.push(i);
        e.2 = e.2.max(g.end - g.start);
    }
    for (by_start, by_end, _) in by_chrom.values_mut() {
        by_start.sort_by_key(|&i| (genes[i].start, i));
        by_end.sort_by_key(|&i| (genes[i].end, i));
    }
    regions
        .par_iter()
        .map(|r| {
            let Some((by_start, by_end, max_len)) = by_chrom.get(r.chrom.as_str()) else {
                return ClosestGene {
                    gene: None,
                    distance: 0,
                    tss_distance: 0,
                };
            };
            // the nearest gene ending at or before the region start, the
            // nearest starting at or after its end, and any that overlap
            let mut best: Option<(u32, usize)> = None;
            let mut take = |d: u32, i: usize| {
                if best.is_none_or(|(bd, bi)| (d, i) < (bd, bi)) {
                    best = Some((d, i));
                }
            };
            let k = by_end.partition_point(|&i| genes[i].end <= r.start);
            if let Some(&i) = k.checked_sub(1).map(|k| &by_end[k]) {
                let end = genes[i].end;
                // every gene with this end is equally near
                for &j in by_end[..k]
                    .iter()
                    .rev()
                    .take_while(|&&j| genes[j].end == end)
                {
                    take(r.start - end, j);
                }
            }
            let k = by_start.partition_point(|&i| genes[i].start < r.end);
            if let Some(&i) = by_start.get(k) {
                let start = genes[i].start;
                for &j in by_start[k..]
                    .iter()
                    .take_while(|&&j| genes[j].start == start)
                {
                    take(start - r.end, j);
                }
            }
            // only genes starting within the longest gene of the region can reach it
            let lo =
                by_start.partition_point(|&i| genes[i].start < r.start.saturating_sub(*max_len));
            for &j in &by_start[lo..k] {
                if genes[j].end > r.start {
                    take(0, j);
                }
            }
            let (distance, i) = best.expect("a chromosome entry holds at least one gene");
            let g = &genes[i];
            let centre = (r.start as i64 + r.end as i64) / 2;
            let offset = centre - g.tss() as i64;
            ClosestGene {
                gene: Some(i),
                distance,
                tss_distance: if g.strand < 0 { -offset } else { offset },
            }
        })
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

/// Regions, gene names and gene ids of the gene activity features, parallel.
pub type GeneActivityRegions = (Vec<FeatureInterval>, Vec<String>, Vec<String>);

/// Gene body plus promoter regions for gene activity (Signac
/// `GeneActivity`).
///
/// Keeps genes of the requested biotypes shorter than `max_width`, extends
/// them strand aware and clips at the chromosome start. Duplicate names are
/// made unique after the filters, so a name keeps its plain form for the
/// gene that survives them; Signac makes them unique before filtering, so
/// it can name that gene `X.1`. Ours matches the RNA gene names better.
///
/// ### Params
///
/// * `genes` - From [`read_gtf_genes`]
/// * `params` - See [`GeneActivityParams`]
///
/// ### Returns
///
/// `(regions, gene names, gene ids)`, parallel, names made unique, or an
/// error when the biotype filter keeps no gene.
pub fn gene_activity_regions(
    genes: &[GeneRecord],
    params: &GeneActivityParams,
) -> Result<GeneActivityRegions, AtacErrors> {
    check_biotypes(genes, &params.biotypes)?;
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
    Ok((regions, make_unique(&names), ids))
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
        let (regions, names, ids) =
            gene_activity_regions(&genes, &GeneActivityParams::default()).unwrap();
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

    #[test]
    fn test_closest_genes() {
        let genes = vec![
            gene("A", 100, 200, 1, ""),
            gene("B", 500, 900, -1, ""),
            gene("C", 520, 530, 1, ""),
        ];
        let r = |s, e| crate::peaks::regions::region("chr1", s, e);
        let got = closest_genes(
            &[r(150, 160), r(300, 320), r(440, 480), r(525, 527)],
            &genes,
        );
        // inside A: 0 bases, centre 155 is 55 downstream of A's TSS at 100
        assert_eq!(
            (got[0].gene, got[0].distance, got[0].tss_distance),
            (Some(0), 0, 55)
        );
        // 100 after A ends, 180 before B starts
        assert_eq!((got[1].gene, got[1].distance), (Some(0), 100));
        // 20 before B; B reads right to left from its TSS at 899, so centre 460
        // lies 439 downstream
        assert_eq!(
            (got[2].gene, got[2].distance, got[2].tss_distance),
            (Some(1), 20, 439)
        );
        // inside both B and C: the first in the list wins
        assert_eq!((got[3].gene, got[3].distance), (Some(1), 0));
        let other = crate::peaks::regions::region("chr9", 0, 10);
        assert_eq!(closest_genes(&[other], &genes)[0].gene, None);
    }
}
