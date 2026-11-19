//! Merge per-sample fragment files into one, as `cellranger-arc aggr` does.
//!
//! Each input's barcodes get a prefix and a suffix, so the same barcode in
//! two samples stays two cells. The output is a sorted BGZF file with a
//! tabix index next to it, which every pass of this crate reads as one
//! experiment. Chromosomes are written in natural order (`chr1`, `chr2`,
//! ..., `chr10`, ..., `chrX`, `chrY`, `chrM`, then the rest by name), each
//! starting a fresh block, so each is one contiguous span of the index.
//!
//! Memory: one chromosome of all inputs at a time, as text, for an input
//! with a tabix index; an input without one is read whole up front.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::Compression as GzLevel;
use flate2::write::{DeflateEncoder, GzEncoder};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::io::fragments::{Parse, chrom_spans, drive, parse_spans};
use crate::prelude::*;

/// Most uncompressed bytes per BGZF block, htslib's choice.
const BGZF_BLOCK_TEXT: usize = 0xff00;

/// One input of [`merge_fragment_files`].
#[derive(Clone, Debug)]
pub struct FragmentInput {
    /// The fragment file.
    pub path: PathBuf,
    /// Put in front of every barcode of this file (`"sample1_"`, say).
    pub prefix: String,
    /// Put after every barcode of this file.
    pub suffix: String,
}

/// What [`merge_fragment_files`] wrote.
#[derive(Clone, Debug)]
pub struct MergeSummary {
    /// Fragments taken from each input, in input order.
    pub n_fragments: Vec<u64>,
    /// Chromosomes of the output, in file order.
    pub chroms: Vec<String>,
}

/// A fragment line ready for the output: sort key, then the renamed line.
type Line = (u32, u32, Vec<u8>);

/// Collects the lines of a file, barcodes renamed, per chromosome.
struct LineCollector<'a> {
    /// Barcode prefix.
    prefix: &'a [u8],
    /// Barcode suffix.
    suffix: &'a [u8],
}

impl Parse<FxHashMap<String, Vec<Line>>> for LineCollector<'_> {
    fn parse(&self, text: &[u8], acc: &mut FxHashMap<String, Vec<Line>>) -> Result<(), AtacErrors> {
        for line in text.split(|&b| b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.is_empty() || line[0] == b'#' {
                continue;
            }
            let bad = || {
                AtacErrors::InvalidArgument(format!(
                    "malformed fragment line '{}'",
                    String::from_utf8_lossy(line)
                ))
            };
            let mut f = line.splitn(5, |&b| b == b'\t');
            let (Some(chrom), Some(start), Some(end), Some(bc)) =
                (f.next(), f.next(), f.next(), f.next())
            else {
                return Err(bad());
            };
            let num = |x: &[u8]| {
                std::str::from_utf8(x)
                    .ok()
                    .and_then(|x| x.parse::<u32>().ok())
            };
            let (Some(s), Some(e)) = (num(start), num(end)) else {
                return Err(bad());
            };
            let mut out =
                Vec::with_capacity(line.len() + self.prefix.len() + self.suffix.len() + 1);
            for part in [chrom, start, end] {
                out.extend_from_slice(part);
                out.push(b'\t');
            }
            out.extend_from_slice(self.prefix);
            out.extend_from_slice(bc);
            out.extend_from_slice(self.suffix);
            if let Some(rest) = f.next() {
                out.push(b'\t');
                out.extend_from_slice(rest);
            }
            out.push(b'\n');
            let chrom = std::str::from_utf8(chrom).map_err(|_| bad())?;
            match acc.get_mut(chrom) {
                Some(v) => v.push((s, e, out)),
                None => {
                    acc.insert(chrom.to_string(), vec![(s, e, out)]);
                }
            }
        }
        Ok(())
    }
}

/// Merge two per-chromosome line maps.
///
/// ### Params
///
/// * `a` - First map
/// * `b` - Second map
///
/// ### Returns
///
/// The union.
fn merge_maps(
    mut a: FxHashMap<String, Vec<Line>>,
    b: FxHashMap<String, Vec<Line>>,
) -> FxHashMap<String, Vec<Line>> {
    for (k, mut v) in b {
        match a.get_mut(&k) {
            Some(x) => x.append(&mut v),
            None => {
                a.insert(k, v);
            }
        }
    }
    a
}

/// Natural chromosome order: numbered ones by number, then X, Y and the
/// mitochondrial genome, then the rest by name.
///
/// ### Params
///
/// * `name` - Chromosome name
///
/// ### Returns
///
/// The sort key.
fn chrom_key(name: &str) -> (u8, u64, String) {
    let bare = name.strip_prefix("chr").unwrap_or(name);
    if let Ok(n) = bare.parse::<u64>() {
        return (0, n, String::new());
    }
    match bare {
        "X" => (1, 0, String::new()),
        "Y" => (1, 1, String::new()),
        "M" | "MT" => (1, 2, String::new()),
        _ => (2, 0, name.to_string()),
    }
}

/// A BGZF writer that starts a fresh block on request and remembers the
/// compressed offset of every block start.
struct BgzfWriter {
    /// The output.
    out: BufWriter<File>,
    /// Compressed bytes written so far.
    written: u64,
}

impl BgzfWriter {
    /// Compress text into whole blocks, in parallel, and write them.
    ///
    /// ### Params
    ///
    /// * `text` - The text; ends at a block boundary afterwards
    ///
    /// ### Returns
    ///
    /// The compressed offset of the first block, or an I/O error.
    fn write_blocks(&mut self, text: &[u8]) -> Result<u64, AtacErrors> {
        let start = self.written;
        let blocks: Vec<Vec<u8>> = text
            .par_chunks(BGZF_BLOCK_TEXT)
            .map(bgzf_block)
            .collect::<Result<_, _>>()?;
        for b in blocks {
            self.out.write_all(&b)?;
            self.written += b.len() as u64;
        }
        Ok(start)
    }

    /// Write the empty end-of-file block and flush.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an I/O error.
    fn finish(mut self) -> Result<(), AtacErrors> {
        self.out.write_all(&bgzf_block(&[])?)?;
        self.out.flush()?;
        Ok(())
    }
}

/// One BGZF block: a gzip member with the `BC` extra field.
///
/// ### Params
///
/// * `text` - At most [`BGZF_BLOCK_TEXT`] bytes
///
/// ### Returns
///
/// The block bytes, or an I/O error from the compressor.
fn bgzf_block(text: &[u8]) -> Result<Vec<u8>, AtacErrors> {
    let mut e = DeflateEncoder::new(Vec::new(), GzLevel::new(6));
    e.write_all(text)?;
    let cdata = e.finish()?;
    let mut crc = flate2::Crc::new();
    crc.update(text);
    let bsize = (18 + cdata.len() + 8 - 1) as u16;
    let mut out = Vec::with_capacity(cdata.len() + 26);
    out.extend_from_slice(&[31, 139, 8, 4, 0, 0, 0, 0, 0, 255, 6, 0, b'B', b'C', 2, 0]);
    out.extend_from_slice(&bsize.to_le_bytes());
    out.extend_from_slice(&cdata);
    out.extend_from_slice(&crc.sum().to_le_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    Ok(out)
}

/// A tabix index (BED preset) with one span per chromosome in the root bin.
///
/// The root bin covers every position, so htslib finds every record of a
/// chromosome through it; region queries scan the chromosome rather than a
/// fine bin, which is all the passes here do anyway.
///
/// ### Params
///
/// * `spans` - `(chromosome, begin, end)` virtual offsets, file order
///
/// ### Returns
///
/// The BGZF-compressed index bytes.
fn tabix_index(spans: &[(String, u64, u64)]) -> Result<Vec<u8>, AtacErrors> {
    let mut idx: Vec<u8> = b"TBI\x01".to_vec();
    let i32s = |v: &mut Vec<u8>, xs: &[i32]| xs.iter().for_each(|x| v.extend(x.to_le_bytes()));
    let names: Vec<u8> = spans
        .iter()
        .flat_map(|s| s.0.bytes().chain(std::iter::once(0)))
        .collect();
    // n_ref, format (generic, 0-based half-open), col_seq, col_beg, col_end,
    // meta char '#', skip, l_nm
    i32s(
        &mut idx,
        &[
            spans.len() as i32,
            0x10000,
            1,
            2,
            3,
            35,
            0,
            names.len() as i32,
        ],
    );
    idx.extend(&names);
    for (_, b, e) in spans {
        // one bin (the root, 0) with one chunk, no linear index
        i32s(&mut idx, &[1, 0, 1]);
        idx.extend(b.to_le_bytes());
        idx.extend(e.to_le_bytes());
        i32s(&mut idx, &[0]);
    }
    let mut gz = GzEncoder::new(Vec::new(), GzLevel::new(6));
    gz.write_all(&idx)?;
    Ok(gz.finish()?)
}

/// Merge fragment files into one sorted, indexed BGZF file.
///
/// ### Params
///
/// * `inputs` - The files and their barcode prefixes and suffixes
/// * `out` - Output path; the index goes to `<out>.tbi`
///
/// ### Returns
///
/// The [`MergeSummary`], or an I/O, parse or argument error. Two inputs
/// that rename a barcode to the same name are not caught: give every input its
/// own prefix or suffix.
pub fn merge_fragment_files(
    inputs: &[FragmentInput],
    out: &Path,
) -> Result<MergeSummary, AtacErrors> {
    if inputs.is_empty() {
        return Err(AtacErrors::InvalidArgument(
            "no fragment files to merge".into(),
        ));
    }
    let collectors: Vec<LineCollector> = inputs
        .iter()
        .map(|i| LineCollector {
            prefix: i.prefix.as_bytes(),
            suffix: i.suffix.as_bytes(),
        })
        .collect();
    let spans: Vec<Option<_>> = inputs
        .iter()
        .map(|i| chrom_spans(&i.path))
        .collect::<Result<_, _>>()?;
    // inputs without an index are read whole now
    let mut loaded: Vec<Option<FxHashMap<String, Vec<Line>>>> = Vec::with_capacity(inputs.len());
    for (i, s) in spans.iter().enumerate() {
        loaded.push(match s {
            Some(_) => None,
            None => Some(drive(
                &inputs[i].path,
                &FxHashMap::default,
                &collectors[i],
                &merge_maps,
            )?),
        });
    }

    let mut chroms: Vec<String> = Vec::new();
    for (s, l) in spans.iter().zip(&loaded) {
        match (s, l) {
            (Some(s), _) => chroms.extend(s.iter().map(|c| c.name.clone())),
            (None, Some(l)) => chroms.extend(l.keys().cloned()),
            (None, None) => unreachable!("an input is either indexed or loaded"),
        }
    }
    chroms.sort_unstable_by_key(|c| chrom_key(c));
    chroms.dedup();

    let mut writer = BgzfWriter {
        out: BufWriter::new(File::create(out)?),
        written: 0,
    };
    writer.write_blocks(
        format!(
            "# merged by bixverse-atac from {} fragment files\n",
            inputs.len()
        )
        .as_bytes(),
    )?;
    let mut n_fragments = vec![0u64; inputs.len()];
    let mut index = Vec::with_capacity(chroms.len());
    for chrom in &chroms {
        let mut lines: Vec<Line> = Vec::new();
        for (i, (s, l)) in spans.iter().zip(loaded.iter_mut()).enumerate() {
            let mut part = match (s, l) {
                (Some(s), _) => match s.iter().find(|c| &c.name == chrom) {
                    Some(span) => parse_spans(
                        &inputs[i].path,
                        std::slice::from_ref(span),
                        &FxHashMap::default,
                        &collectors[i],
                        &merge_maps,
                    )?
                    .remove(chrom)
                    .unwrap_or_default(),
                    None => Vec::new(),
                },
                (None, Some(l)) => l.remove(chrom).unwrap_or_default(),
                (None, None) => unreachable!("an input is either indexed or loaded"),
            };
            n_fragments[i] += part.len() as u64;
            lines.append(&mut part);
        }
        if lines.is_empty() {
            continue;
        }
        lines.par_sort_unstable_by(|a, b| (a.0, a.1, &a.2).cmp(&(b.0, b.1, &b.2)));
        let text: Vec<u8> = lines.into_iter().flat_map(|l| l.2).collect();
        let beg = writer.write_blocks(&text)?;
        index.push((chrom.clone(), beg << 16, writer.written << 16));
    }
    writer.finish()?;

    let mut tbi = out.as_os_str().to_owned();
    tbi.push(".tbi");
    std::fs::write(PathBuf::from(tbi), tabix_index(&index)?)?;
    Ok(MergeSummary {
        n_fragments,
        chroms: index.into_iter().map(|s| s.0).collect(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fragments::{BarcodeIndex, Fragment, fold_fragments};

    /// Write fragment text to `dir/name`, BGZF with a tabix index when asked.
    fn write_input(
        dir: &Path,
        name: &str,
        lines: &[(&str, u32, u32, &str)],
        indexed: bool,
    ) -> PathBuf {
        let p = dir.join(name);
        if indexed {
            // reuse the merge itself to write an indexed single file
            let plain = dir.join(format!("{name}.plain"));
            let text: String = lines
                .iter()
                .map(|(c, s, e, b)| format!("{c}\t{s}\t{e}\t{b}\t1\n"))
                .collect();
            std::fs::write(&plain, text).unwrap();
            merge_fragment_files(
                &[FragmentInput {
                    path: plain,
                    prefix: String::new(),
                    suffix: String::new(),
                }],
                &p,
            )
            .unwrap();
        } else {
            let text: String = lines
                .iter()
                .map(|(c, s, e, b)| format!("{c}\t{s}\t{e}\t{b}\t1\n"))
                .collect();
            std::fs::write(&p, text).unwrap();
        }
        p
    }

    #[test]
    fn test_merge_renames_sorts_and_indexes() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = write_input(
            dir.path(),
            "a.tsv.gz",
            &[
                ("chr1", 10, 50, "AAAC-1"),
                ("chr10", 5, 9, "AAAC-1"),
                ("chr2", 7, 30, "CCCG-1"),
            ],
            true,
        );
        let b = write_input(
            dir.path(),
            "b.tsv",
            &[
                ("chr2", 3, 40, "AAAC-1"),
                ("chr1", 20, 60, "AAAC-1"),
                ("chrX", 1, 5, "GGGT-1"),
            ],
            false,
        );
        let out = dir.path().join("merged.tsv.gz");
        let summary = merge_fragment_files(
            &[
                FragmentInput {
                    path: a,
                    prefix: "s1_".into(),
                    suffix: String::new(),
                },
                FragmentInput {
                    path: b,
                    prefix: String::new(),
                    suffix: "_s2".into(),
                },
            ],
            &out,
        )
        .unwrap();
        assert_eq!(summary.n_fragments, vec![3, 3]);
        assert_eq!(summary.chroms, ["chr1", "chr2", "chr10", "chrX"]);

        // the index reads back as one span per chromosome, in file order
        let spans = chrom_spans(&out).unwrap().unwrap();
        let names: Vec<&str> = spans.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["chr1", "chr2", "chr10", "chrX"]);

        let cells: Vec<String> = ["s1_AAAC-1", "AAAC-1_s2", "s1_CCCG-1", "GGGT-1_s2"]
            .map(String::from)
            .to_vec();
        let idx = BarcodeIndex::new(&cells).unwrap();
        let mut got = fold_fragments(
            &out,
            &idx,
            Vec::new,
            |acc: &mut Vec<(String, Fragment)>, chrom, frags| {
                acc.extend(frags.iter().map(|f| (chrom.to_string(), *f)))
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        )
        .unwrap();
        got.sort_by(|a, b| (&a.0, a.1.start).cmp(&(&b.0, b.1.start)));
        let flat: Vec<(&str, u32, u32)> = got
            .iter()
            .map(|(c, f)| (c.as_str(), f.start, f.cell))
            .collect();
        assert_eq!(
            flat,
            [
                ("chr1", 10, 0),
                ("chr1", 20, 1),
                ("chr10", 5, 0),
                ("chr2", 3, 1),
                ("chr2", 7, 2),
                ("chrX", 1, 3)
            ]
        );
        // the merged file is sorted within each chromosome
        let text = {
            let mut s = String::new();
            std::io::Read::read_to_string(
                &mut flate2::read::MultiGzDecoder::new(File::open(&out).unwrap()),
                &mut s,
            )
            .unwrap();
            s
        };
        let chr1: Vec<&str> = text.lines().filter(|l| l.starts_with("chr1\t")).collect();
        assert_eq!(
            chr1,
            ["chr1\t10\t50\ts1_AAAC-1\t1", "chr1\t20\t60\tAAAC-1_s2\t1"]
        );
    }
}
