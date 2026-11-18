//! Fragment files: parallel reading of 10x style `fragments.tsv.gz`.
//!
//! A fragment file is BED-like, one fragment per line: chromosome, start
//! (0-based), end (exclusive), barcode and, optionally, the number of read
//! pairs. Lines starting with `#` are headers. Files are coordinate sorted
//! and BGZF compressed: a run of independent gzip members of at most 64 KiB
//! each, with the compressed size in the gzip extra field. That makes them
//! splittable, so a full pass maps the file, indexes the blocks, and inflates
//! and parses groups of blocks in parallel. A line cut by a group boundary is
//! stitched back together after the parallel pass. Plain gzip (not
//! splittable) inflates on one thread and parses in parallel; uncompressed
//! files split at newlines.
//!
//! Every pass is a fold: each worker owns an accumulator and gets runs of
//! fragments on one chromosome, and the accumulators are merged at the end.
//! Only fragments of the requested barcodes reach the fold; everything else
//! is dropped while parsing.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use flate2::read::{GzDecoder, MultiGzDecoder};
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;

/// BGZF blocks inflated per parallel task. Blocks hold up to 64 KiB of text,
/// so a group is a few MiB: large enough that the boundary stitching and the
/// task overhead vanish, small enough to balance the threads.
const BGZF_BLOCKS_PER_GROUP: usize = 64;

/// Text per task on the plain gzip and uncompressed paths.
const TEXT_CHUNK_BYTES: usize = 4 << 20;

/// Bytes of a BGZF block header up to and including the extra field length.
const GZIP_FIXED_HEADER: usize = 12;

/// Partial lines at the edges of a BGZF group: `(segment, group, head, tail)`.
type GroupEdges = (usize, usize, Vec<u8>, Vec<u8>);

///////////
// Types //
///////////

/// One fragment of a requested cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fragment {
    /// 0-based start, the left Tn5 insertion.
    pub start: u32,
    /// End as written (exclusive in BED terms). Cell Ranger and Signac take
    /// it as the right insertion as is.
    pub end: u32,
    /// Position of the barcode in the requested list.
    pub cell: u32,
    /// Read pairs supporting the fragment (column 5), 1 when absent.
    pub count: u32,
}

impl Fragment {
    /// Fragment length in base pairs.
    ///
    /// ### Returns
    ///
    /// `end - start`.
    #[inline]
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Whether the fragment is empty (never true for a valid record).
    ///
    /// ### Returns
    ///
    /// `end == start`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// Lookup from barcode bytes to the position in the requested list.
pub struct BarcodeIndex<'a> {
    /// Barcode to position.
    map: FxHashMap<&'a [u8], u32>,
}

impl<'a> BarcodeIndex<'a> {
    /// Index a list of barcodes.
    ///
    /// ### Params
    ///
    /// * `barcodes` - The cells, in the order the passes report them
    ///
    /// ### Returns
    ///
    /// The index, or an error for a duplicated barcode or more than
    /// `u32::MAX` cells.
    pub fn new(barcodes: &'a [String]) -> Result<Self, AtacErrors> {
        if barcodes.len() > u32::MAX as usize {
            return Err(AtacErrors::InvalidArgument(
                "more barcodes than fit u32".into(),
            ));
        }
        let mut map = FxHashMap::default();
        map.reserve(barcodes.len());
        for (i, b) in barcodes.iter().enumerate() {
            if map.insert(b.as_bytes(), i as u32).is_some() {
                return Err(AtacErrors::InvalidArgument(format!(
                    "barcode '{b}' is listed twice"
                )));
            }
        }
        Ok(Self { map })
    }

    /// Number of barcodes.
    ///
    /// ### Returns
    ///
    /// The count.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no barcode is indexed.
    ///
    /// ### Returns
    ///
    /// `true` for an empty index.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Position of a barcode.
    ///
    /// ### Params
    ///
    /// * `b` - Barcode bytes
    ///
    /// ### Returns
    ///
    /// The position, or `None` for a barcode not requested.
    #[inline]
    fn get(&self, b: &[u8]) -> Option<u32> {
        self.map.get(b).copied()
    }
}

/// How a fragment file is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compression {
    /// Block gzip, splittable.
    Bgzf,
    /// Plain gzip, one stream.
    Gzip,
    /// Text.
    None,
}

///////////////////
// Line parsing  //
///////////////////

/// Parse an unsigned decimal field.
///
/// ### Params
///
/// * `s` - The bytes
///
/// ### Returns
///
/// The value, or `None` for an empty field, a non-digit or overflow.
#[inline]
fn parse_u32(s: &[u8]) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut v: u32 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as u32)?;
    }
    Some(v)
}

/// A malformed line, for the error message.
///
/// ### Params
///
/// * `line` - The line
///
/// ### Returns
///
/// The error.
fn bad_line(line: &[u8]) -> AtacErrors {
    AtacErrors::InvalidArgument(format!(
        "malformed fragment line '{}'",
        String::from_utf8_lossy(line)
    ))
}

/// Parse complete lines and feed runs of one chromosome to the fold.
///
/// ### Params
///
/// * `text` - Whole lines, `\n` separated; a missing final newline is fine
/// * `barcodes` - The requested cells
/// * `acc` - The worker's accumulator
/// * `fold` - Consumer of `(accumulator, chromosome, fragments)`
/// * `buf` - Scratch for the current run
///
/// ### Returns
///
/// `Ok(())`, or an error for a malformed line.
fn parse_lines<A, F>(
    text: &[u8],
    barcodes: &BarcodeIndex,
    acc: &mut A,
    fold: &F,
    buf: &mut Vec<Fragment>,
) -> Result<(), AtacErrors>
where
    F: Fn(&mut A, &str, &[Fragment]),
{
    buf.clear();
    let mut run_chrom: &[u8] = &[];
    for line in text.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        let mut fields = line.split(|&b| b == b'\t');
        let (Some(chrom), Some(start), Some(end), Some(bc)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(bad_line(line));
        };
        let Some(cell) = barcodes.get(bc) else {
            continue;
        };
        let start = parse_u32(start).ok_or_else(|| bad_line(line))?;
        let end = parse_u32(end).ok_or_else(|| bad_line(line))?;
        if end <= start {
            return Err(bad_line(line));
        }
        let count = match fields.next() {
            Some(c) => parse_u32(c).ok_or_else(|| bad_line(line))?,
            None => 1,
        };
        if chrom != run_chrom {
            if !buf.is_empty() {
                fold(acc, chrom_str(run_chrom, line)?, buf);
                buf.clear();
            }
            run_chrom = chrom;
        }
        buf.push(Fragment {
            start,
            end,
            cell,
            count,
        });
    }
    if !buf.is_empty() {
        fold(acc, chrom_str(run_chrom, run_chrom)?, buf);
        buf.clear();
    }
    Ok(())
}

/// Chromosome bytes as UTF-8.
///
/// ### Params
///
/// * `chrom` - The bytes
/// * `line` - Context for the error
///
/// ### Returns
///
/// The name, or an error for invalid UTF-8.
fn chrom_str<'a>(chrom: &'a [u8], line: &[u8]) -> Result<&'a str, AtacErrors> {
    std::str::from_utf8(chrom).map_err(|_| bad_line(line))
}

//////////
// BGZF //
//////////

/// Size of the BGZF block at the start of `bytes`.
///
/// ### Params
///
/// * `bytes` - File bytes from a block start
///
/// ### Returns
///
/// The total block length, or `None` when the bytes do not start a BGZF
/// block (plain gzip has no `BC` extra subfield).
fn bgzf_block_len(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < GZIP_FIXED_HEADER || bytes[0] != 31 || bytes[1] != 139 || bytes[2] != 8 {
        return None;
    }
    if bytes[3] & 4 == 0 {
        return None;
    }
    let xlen = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
    let extra = bytes.get(GZIP_FIXED_HEADER..GZIP_FIXED_HEADER + xlen)?;
    let mut i = 0;
    while i + 4 <= extra.len() {
        let slen = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
        if extra[i] == b'B' && extra[i + 1] == b'C' && slen == 2 {
            let bsize = u16::from_le_bytes([*extra.get(i + 4)?, *extra.get(i + 5)?]) as usize;
            return Some(bsize + 1);
        }
        i += 4 + slen;
    }
    None
}

/// Offsets of every BGZF block.
///
/// ### Params
///
/// * `bytes` - The whole file
///
/// ### Returns
///
/// Block start offsets plus the file length as a final sentinel, or an error
/// for a truncated or non-BGZF block.
fn bgzf_block_offsets(bytes: &[u8]) -> Result<Vec<usize>, AtacErrors> {
    let mut offsets = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let len = bgzf_block_len(&bytes[pos..]).ok_or_else(|| {
            AtacErrors::InvalidArgument(format!("BGZF block expected at byte {pos}"))
        })?;
        if pos + len > bytes.len() {
            return Err(AtacErrors::InvalidArgument("truncated BGZF file".into()));
        }
        offsets.push(pos);
        pos += len;
    }
    offsets.push(bytes.len());
    Ok(offsets)
}

/// Detect the compression from the first bytes.
///
/// ### Params
///
/// * `bytes` - The whole file
///
/// ### Returns
///
/// The [`Compression`].
fn detect_compression(bytes: &[u8]) -> Compression {
    if bytes.len() >= 2 && bytes[0] == 31 && bytes[1] == 139 {
        if bgzf_block_len(bytes).is_some() {
            Compression::Bgzf
        } else {
            Compression::Gzip
        }
    } else {
        Compression::None
    }
}

///////////////
// The fold  //
///////////////

/// Complete lines of a chunk plus the partial lines at its edges.
///
/// ### Params
///
/// * `text` - The chunk
/// * `first` - Whether the chunk starts the file (no partial head)
///
/// ### Returns
///
/// `(head, body, tail)`: the bytes up to and including the first newline
/// (empty for the first chunk), the complete lines after it, and the bytes
/// after the last newline.
fn split_edges(text: &[u8], first: bool) -> (&[u8], &[u8], &[u8]) {
    let head_end = if first {
        0
    } else {
        match text.iter().position(|&b| b == b'\n') {
            Some(i) => i + 1,
            None => return (text, &[], &[]),
        }
    };
    let tail_start = match text[head_end..].iter().rposition(|&b| b == b'\n') {
        Some(i) => head_end + i + 1,
        None => head_end,
    };
    (
        &text[..head_end],
        &text[head_end..tail_start],
        &text[tail_start..],
    )
}

/// Fold over the fragments of the requested cells.
///
/// Each worker starts from `init()`, receives runs of fragments on one
/// chromosome through `fold`, and the workers' accumulators are combined with
/// `reduce`. Runs arrive in no particular order across workers, but within a
/// run fragments keep file order (sorted by start).
///
/// ### Params
///
/// * `path` - The fragment file (BGZF, gzip or text)
/// * `barcodes` - Cells to keep; other barcodes are skipped
/// * `init` - New accumulator
/// * `fold` - Consumer of `(accumulator, chromosome, fragments)`
/// * `reduce` - Merge of two accumulators
///
/// ### Returns
///
/// The combined accumulator, or an I/O or parse error.
pub fn fold_fragments<A, I, F, R>(
    path: &Path,
    barcodes: &BarcodeIndex,
    init: I,
    fold: F,
    reduce: R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        return Ok(init());
    }
    // SAFETY: the file is opened read-only and not modified while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    match detect_compression(bytes) {
        Compression::Bgzf => fold_bgzf(bytes, barcodes, &init, &fold, &reduce),
        Compression::None => fold_text(bytes, barcodes, &init, &fold, &reduce),
        Compression::Gzip => fold_gzip(&file, barcodes, &init, &fold, &reduce),
    }
}

///////////
// Tabix //
///////////

/// Tabix pseudo-bin holding per-reference metadata rather than records.
const TABIX_META_BIN: u32 = 37450;

/// Where one chromosome lies in a BGZF fragment file, from its tabix index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChromSpan {
    /// Chromosome name.
    pub name: String,
    /// Virtual offset of its first record.
    beg: u64,
    /// Virtual offset just past its last record.
    end: u64,
}

impl ChromSpan {
    /// Compressed bytes the chromosome spans, a proxy for its fragment count.
    ///
    /// ### Returns
    ///
    /// The byte count.
    pub fn compressed_bytes(&self) -> u64 {
        (self.end >> 16).saturating_sub(self.beg >> 16)
    }
}

/// Little-endian reader over the inflated index.
struct IndexCursor<'a> {
    /// The inflated index.
    bytes: &'a [u8],
    /// Read position.
    pos: usize,
}

impl IndexCursor<'_> {
    /// Take `n` bytes.
    ///
    /// ### Params
    ///
    /// * `n` - Byte count
    ///
    /// ### Returns
    ///
    /// The bytes, or an error past the end.
    fn take(&mut self, n: usize) -> Result<&[u8], AtacErrors> {
        let out = self
            .bytes
            .get(self.pos..self.pos + n)
            .ok_or_else(|| AtacErrors::InvalidArgument("truncated tabix index".into()))?;
        self.pos += n;
        Ok(out)
    }

    /// Read an `i32` used as a count.
    ///
    /// ### Returns
    ///
    /// The value, or an error for a negative count.
    fn count(&mut self) -> Result<usize, AtacErrors> {
        let v = i32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes"));
        usize::try_from(v)
            .map_err(|_| AtacErrors::InvalidArgument("negative count in tabix index".into()))
    }

    /// Read a `u32`.
    ///
    /// ### Returns
    ///
    /// The value.
    fn u32(&mut self) -> Result<u32, AtacErrors> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    /// Read a `u64`.
    ///
    /// ### Returns
    ///
    /// The value.
    fn u64(&mut self) -> Result<u64, AtacErrors> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
}

/// Parse a tabix index into chromosome spans.
///
/// ### Params
///
/// * `raw` - The `.tbi` file bytes (BGZF compressed)
///
/// ### Returns
///
/// The spans of chromosomes with records, in file order, or an error for a
/// malformed index.
fn parse_tabix(raw: &[u8]) -> Result<Vec<ChromSpan>, AtacErrors> {
    let mut bytes = Vec::new();
    MultiGzDecoder::new(raw).read_to_end(&mut bytes)?;
    let mut c = IndexCursor {
        bytes: &bytes,
        pos: 0,
    };
    if c.take(4)? != b"TBI\x01" {
        return Err(AtacErrors::InvalidArgument("not a tabix index".into()));
    }
    let n_ref = c.count()?;
    // format, col_seq, col_beg, col_end, meta, skip
    c.take(24)?;
    let l_nm = c.count()?;
    let names: Vec<String> = c
        .take(l_nm)?
        .split(|&b| b == 0)
        .filter(|n| !n.is_empty())
        .map(|n| String::from_utf8_lossy(n).into_owned())
        .collect();
    if names.len() != n_ref {
        return Err(AtacErrors::InvalidArgument(
            "tabix index names do not match its reference count".into(),
        ));
    }

    let mut spans = Vec::new();
    for name in names {
        let (mut beg, mut end) = (u64::MAX, 0u64);
        for _ in 0..c.count()? {
            let bin = c.u32()?;
            for _ in 0..c.count()? {
                let (b, e) = (c.u64()?, c.u64()?);
                if bin != TABIX_META_BIN {
                    beg = beg.min(b);
                    end = end.max(e);
                }
            }
        }
        let n_intv = c.count()?;
        c.take(8 * n_intv)?;
        if beg < end {
            spans.push(ChromSpan { name, beg, end });
        }
    }
    spans.sort_unstable_by_key(|s| s.beg);
    Ok(spans)
}

/// The chromosomes of a BGZF fragment file from its tabix index
/// (`<path>.tbi`), so that passes can read one chromosome at a time.
///
/// ### Params
///
/// * `path` - The fragment file
///
/// ### Returns
///
/// The spans in file order; `None` when the file is not BGZF or has no
/// index next to it. An error for an unreadable or malformed index.
pub fn chrom_spans(path: &Path) -> Result<Option<Vec<ChromSpan>>, AtacErrors> {
    let mut tbi = path.as_os_str().to_owned();
    tbi.push(".tbi");
    let tbi = Path::new(&tbi);
    if !tbi.exists() {
        return Ok(None);
    }
    let mut head = [0u8; GZIP_FIXED_HEADER + 6];
    let n = File::open(path)?.read(&mut head)?;
    if detect_compression(&head[..n]) != Compression::Bgzf {
        return Ok(None);
    }
    parse_tabix(&std::fs::read(tbi)?).map(Some)
}

/// Fold over the fragments of some chromosomes only, read through their
/// tabix spans. Same contract as [`fold_fragments`].
///
/// ### Params
///
/// * `path` - The BGZF fragment file the spans come from
/// * `spans` - Chromosomes to read, from [`chrom_spans`]
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer of `(accumulator, chromosome, fragments)`
/// * `reduce` - Merge of two accumulators
///
/// ### Returns
///
/// The combined accumulator, or an I/O or parse error, or an error for an
/// index that does not match the file.
pub fn fold_fragment_spans<A, I, F, R>(
    path: &Path,
    spans: &[ChromSpan],
    barcodes: &BarcodeIndex,
    init: I,
    fold: F,
    reduce: R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    let file = File::open(path)?;
    // SAFETY: the file is opened read-only and not modified while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;
    // only the spans' own block headers are walked, so pages elsewhere in
    // the file are never touched
    let stale = || AtacErrors::InvalidArgument("the tabix index does not match the file".into());
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut segments = Vec::with_capacity(spans.len());
    for s in spans {
        let (first, last) = ((s.beg >> 16) as usize, (s.end >> 16) as usize);
        let end_in = (s.end & 0xffff) as usize;
        let b0 = blocks.len();
        let mut pos = first;
        // the end offset points into block `last`, or to its start when the
        // span ends with the block before
        while pos < last || (pos == last && end_in > 0) {
            let len = bytes
                .get(pos..)
                .and_then(bgzf_block_len)
                .filter(|&l| pos + l <= bytes.len())
                .ok_or_else(stale)?;
            blocks.push((pos, pos + len));
            pos += len;
        }
        let lands = if end_in > 0 {
            blocks.last().map(|b| b.0) == Some(last)
        } else {
            pos == last
        };
        if !lands || blocks.len() == b0 {
            return Err(stale());
        }
        segments.push(BlockSegment {
            b0,
            b1: blocks.len(),
            skip: (s.beg & 0xffff) as usize,
            end: (end_in > 0).then_some(end_in),
        });
    }
    fold_bgzf_segments(bytes, &blocks, &segments, barcodes, &init, &fold, &reduce)
}

/// One read of a multi-pass method: some chromosomes through the index, or
/// the whole file (`None`).
pub(crate) type FragmentBatch = Option<Vec<ChromSpan>>;

/// Plan the reads of a pass: chromosome batches under a memory budget when
/// the file has a tabix index, the whole file otherwise.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `max_mem_gb` - Budget, `None` for one chromosome per batch
/// * `mem_per_compressed_byte` - The caller's memory per compressed byte
///   read, to turn the budget into bytes
/// * `keep` - Chromosomes to read; the rest are skipped (only possible with
///   an index)
///
/// ### Returns
///
/// The batches, and the chromosomes in file order (only the first one when
/// there is no index), or an I/O or index error.
pub(crate) fn plan_batches(
    path: &Path,
    max_mem_gb: Option<f64>,
    mem_per_compressed_byte: f64,
    keep: impl Fn(&str) -> bool,
) -> Result<(Vec<FragmentBatch>, Vec<String>), AtacErrors> {
    let Some(spans) = chrom_spans(path)? else {
        let order = first_chromosome(path)?.into_iter().collect();
        return Ok((vec![None], order));
    };
    let order = spans.iter().map(|s| s.name.clone()).collect();
    let cap = max_mem_gb.map_or(0.0, |gb| gb * 1e9 / mem_per_compressed_byte);
    let mut batches: Vec<Vec<ChromSpan>> = Vec::new();
    let mut size = 0.0;
    for span in spans.into_iter().filter(|s| keep(&s.name)) {
        let bytes = span.compressed_bytes() as f64;
        match batches.last_mut() {
            Some(b) if size + bytes <= cap => b.push(span),
            _ => {
                size = 0.0;
                batches.push(vec![span]);
            }
        }
        size += bytes;
    }
    Ok((batches.into_iter().map(Some).collect(), order))
}

/// [`fold_fragment_spans`] for a batch, [`fold_fragments`] for the whole
/// file.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `batch` - From [`plan_batches`]
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer of `(accumulator, chromosome, fragments)`
/// * `reduce` - Merge of two accumulators
///
/// ### Returns
///
/// The combined accumulator, or an I/O or parse error.
pub(crate) fn fold_batch<A, I, F, R>(
    path: &Path,
    batch: &FragmentBatch,
    barcodes: &BarcodeIndex,
    init: I,
    fold: F,
    reduce: R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    match batch {
        Some(spans) => fold_fragment_spans(path, spans, barcodes, init, fold, reduce),
        None => fold_fragments(path, barcodes, init, fold, reduce),
    }
}

/// The chromosome of the first fragment line of a file.
///
/// ### Params
///
/// * `path` - Fragment file, gzip, BGZF or plain
///
/// ### Returns
///
/// The chromosome, `None` for a file without fragment lines, or an I/O
/// error.
pub(crate) fn first_chromosome(path: &Path) -> Result<Option<String>, AtacErrors> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 2];
    let gz = file.read(&mut magic)? == 2 && magic == [0x1f, 0x8b];
    file.seek(SeekFrom::Start(0))?;
    let reader: Box<dyn BufRead> = if gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        return Ok(line.split('\t').next().map(str::to_string));
    }
    Ok(None)
}

/// A stretch of a BGZF file to parse: blocks `[b0, b1)`, minus `skip` bytes
/// at the start of the first block's text, cut after `end` bytes of the last
/// block's text when set.
#[derive(Clone, Copy, Debug)]
struct BlockSegment {
    /// First block.
    b0: usize,
    /// One past the last block.
    b1: usize,
    /// Text bytes of block `b0` before the segment starts.
    skip: usize,
    /// Text bytes of block `b1 - 1` that belong to the segment; `None` for
    /// all of them.
    end: Option<usize>,
}

/// [`fold_fragments`] on a BGZF file.
///
/// ### Params
///
/// * `bytes` - The mapped file
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer
/// * `reduce` - Merge
///
/// ### Returns
///
/// The combined accumulator.
fn fold_bgzf<A, I, F, R>(
    bytes: &[u8],
    barcodes: &BarcodeIndex,
    init: &I,
    fold: &F,
    reduce: &R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    let offsets = bgzf_block_offsets(bytes)?;
    let blocks: Vec<(usize, usize)> = offsets.windows(2).map(|w| (w[0], w[1])).collect();
    let whole = BlockSegment {
        b0: 0,
        b1: blocks.len(),
        skip: 0,
        end: None,
    };
    fold_bgzf_segments(bytes, &blocks, &[whole], barcodes, init, fold, reduce)
}

/// Fold over BGZF segments that each start and end at a line boundary.
///
/// Segments are cut into groups of [`BGZF_BLOCKS_PER_GROUP`] blocks, inflated
/// and parsed in parallel; lines cut by a group boundary are stitched back
/// within their segment afterwards.
///
/// ### Params
///
/// * `bytes` - The mapped file
/// * `blocks` - Byte ranges of the blocks the segments index into
/// * `segments` - The stretches to read
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer
/// * `reduce` - Merge
///
/// ### Returns
///
/// The combined accumulator.
fn fold_bgzf_segments<A, I, F, R>(
    bytes: &[u8],
    blocks: &[(usize, usize)],
    segments: &[BlockSegment],
    barcodes: &BarcodeIndex,
    init: &I,
    fold: &F,
    reduce: &R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    // (segment, group in segment, first block, end block)
    let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (s, seg) in segments.iter().enumerate() {
        for (g, b) in (seg.b0..seg.b1).step_by(BGZF_BLOCKS_PER_GROUP).enumerate() {
            groups.push((s, g, b, (b + BGZF_BLOCKS_PER_GROUP).min(seg.b1)));
        }
    }
    let edges: Mutex<Vec<GroupEdges>> = Mutex::new(Vec::new());

    let acc = groups
        .par_iter()
        .try_fold(
            || (init(), Vec::new(), Vec::new()),
            |(mut acc, mut text, mut buf), &(s, g, b0, b1)| {
                let seg = segments[s];
                text.clear();
                let mut last_start = 0;
                for &(from, to) in &blocks[b0..b1] {
                    last_start = text.len();
                    GzDecoder::new(&bytes[from..to]).read_to_end(&mut text)?;
                }
                if b1 == seg.b1
                    && let Some(end) = seg.end
                {
                    text.truncate(last_start + end);
                }
                let skip = if g == 0 { seg.skip.min(text.len()) } else { 0 };
                let (head, body, tail) = split_edges(&text[skip..], g == 0);
                parse_lines(body, barcodes, &mut acc, fold, &mut buf)?;
                edges.lock().expect("edge list lock poisoned").push((
                    s,
                    g,
                    head.to_vec(),
                    tail.to_vec(),
                ));
                Ok::<_, AtacErrors>((acc, text, buf))
            },
        )
        .map(|r| r.map(|(acc, _, _)| acc))
        .try_reduce(init, |a, b| Ok(reduce(a, b)))?;

    // a line cut at a group boundary is the tail of one group plus the head
    // of the next one of the same segment
    let mut edges = edges.into_inner().expect("edge list lock poisoned");
    edges.sort_unstable_by_key(|e| (e.0, e.1));
    let mut stitched = Vec::new();
    for w in 0..edges.len() {
        stitched.extend_from_slice(&edges[w].3);
        if let Some(next) = edges.get(w + 1).filter(|n| n.0 == edges[w].0) {
            stitched.extend_from_slice(&next.2);
        }
        if !stitched.is_empty() && !stitched.ends_with(b"\n") {
            stitched.push(b'\n');
        }
    }
    let mut rest = init();
    parse_lines(&stitched, barcodes, &mut rest, fold, &mut Vec::new())?;
    Ok(reduce(acc, rest))
}

/// [`fold_fragments`] on an uncompressed file.
///
/// ### Params
///
/// * `bytes` - The mapped file
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer
/// * `reduce` - Merge
///
/// ### Returns
///
/// The combined accumulator.
fn fold_text<A, I, F, R>(
    bytes: &[u8],
    barcodes: &BarcodeIndex,
    init: &I,
    fold: &F,
    reduce: &R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = (start + TEXT_CHUNK_BYTES).min(bytes.len());
        while end < bytes.len() && bytes[end - 1] != b'\n' {
            end += 1;
        }
        chunks.push(&bytes[start..end]);
        start = end;
    }
    chunks
        .par_iter()
        .try_fold(
            || (init(), Vec::new()),
            |(mut acc, mut buf), chunk| {
                parse_lines(chunk, barcodes, &mut acc, fold, &mut buf)?;
                Ok::<_, AtacErrors>((acc, buf))
            },
        )
        .map(|r| r.map(|(acc, _)| acc))
        .try_reduce(init, |a, b| Ok(reduce(a, b)))
}

/// [`fold_fragments`] on a plain gzip file: one inflating thread, parallel
/// parsing.
///
/// ### Params
///
/// * `file` - The open file
/// * `barcodes` - Cells to keep
/// * `init` - New accumulator
/// * `fold` - Consumer
/// * `reduce` - Merge
///
/// ### Returns
///
/// The combined accumulator.
fn fold_gzip<A, I, F, R>(
    file: &File,
    barcodes: &BarcodeIndex,
    init: &I,
    fold: &F,
    reduce: &R,
) -> Result<A, AtacErrors>
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, &str, &[Fragment]) + Sync + Send,
    R: Fn(A, A) -> A + Sync + Send,
{
    let mut decoder = MultiGzDecoder::new(std::io::BufReader::new(file));
    let mut carry: Vec<u8> = Vec::new();
    let mut acc = init();
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let batch = rayon::current_num_threads().max(1) * 2;
    let mut eof = false;

    while !eof {
        // inflate a batch of chunks, each cut after its last newline
        while chunks.len() < batch {
            let mut chunk = std::mem::take(&mut carry);
            let have = chunk.len();
            chunk.resize(have + TEXT_CHUNK_BYTES, 0);
            let mut filled = have;
            while filled < chunk.len() {
                let n = decoder.read(&mut chunk[filled..])?;
                if n == 0 {
                    eof = true;
                    break;
                }
                filled += n;
            }
            chunk.truncate(filled);
            if !eof && let Some(nl) = chunk.iter().rposition(|&b| b == b'\n') {
                carry = chunk.split_off(nl + 1);
            }
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
            if eof {
                break;
            }
        }
        let part = chunks
            .par_iter()
            .try_fold(
                || (init(), Vec::new()),
                |(mut acc, mut buf), chunk| {
                    parse_lines(chunk, barcodes, &mut acc, fold, &mut buf)?;
                    Ok::<_, AtacErrors>((acc, buf))
                },
            )
            .map(|r| r.map(|(acc, _)| acc))
            .try_reduce(init, |a, b| Ok(reduce(a, b)))?;
        acc = reduce(acc, part);
        chunks.clear();
    }
    Ok(acc)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression as GzLevel;
    use flate2::write::GzEncoder;
    use std::io::Write;

    /// Synthetic fragment file text: three chromosomes, four barcodes, one
    /// header line.
    ///
    /// ### Params
    ///
    /// * `n` - Fragments per chromosome
    ///
    /// ### Returns
    ///
    /// The text.
    fn synthetic_text(n: u32) -> String {
        let mut s = String::from("# id=test\n");
        for chrom in ["chr1", "chr2", "chrX"] {
            for i in 0..n {
                let start = i * 37;
                let end = start + 50 + (i % 400);
                let bc = ["AAAC-1", "CCCG-1", "GGGT-1", "TTTA-1"][(i % 4) as usize];
                s.push_str(&format!("{chrom}\t{start}\t{end}\t{bc}\t{}\n", 1 + i % 3));
            }
        }
        s
    }

    /// Write `text` as a BGZF file: gzip members of at most `block` bytes of
    /// text each, with the BC extra field.
    ///
    /// ### Params
    ///
    /// * `text` - The content
    /// * `block` - Uncompressed bytes per block
    ///
    /// ### Returns
    ///
    /// The file bytes.
    fn bgzf_bytes(text: &[u8], block: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in text.chunks(block).chain(std::iter::once(&[][..])) {
            let mut e = flate2::write::DeflateEncoder::new(Vec::new(), GzLevel::fast());
            e.write_all(piece).unwrap();
            let cdata = e.finish().unwrap();
            let mut crc = flate2::Crc::new();
            crc.update(piece);
            let bsize = (18 + cdata.len() + 8 - 1) as u16;
            out.extend_from_slice(&[31, 139, 8, 4, 0, 0, 0, 0, 0, 255, 6, 0, b'B', b'C', 2, 0]);
            out.extend_from_slice(&bsize.to_le_bytes());
            out.extend_from_slice(&cdata);
            out.extend_from_slice(&crc.sum().to_le_bytes());
            out.extend_from_slice(&(piece.len() as u32).to_le_bytes());
        }
        out
    }

    /// Every fragment of the requested cells, sorted, per chromosome.
    ///
    /// ### Params
    ///
    /// * `path` - The file
    /// * `cells` - Requested barcodes
    ///
    /// ### Returns
    ///
    /// `(chrom, fragment)` pairs, sorted.
    fn collect(path: &Path, cells: &[String]) -> Vec<(String, Fragment)> {
        let idx = BarcodeIndex::new(cells).unwrap();
        let mut v = fold_fragments(
            path,
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
        v.sort_by(|a, b| (&a.0, a.1.start, a.1.cell).cmp(&(&b.0, b.1.start, b.1.cell)));
        v
    }

    /// Scratch file removed on drop.
    struct TempPath(std::path::PathBuf);

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Write bytes to a scratch file.
    ///
    /// ### Params
    ///
    /// * `tag` - Unique name part
    /// * `bytes` - Content
    ///
    /// ### Returns
    ///
    /// The guard.
    fn temp_file(tag: &str, bytes: &[u8]) -> TempPath {
        let p = std::env::temp_dir().join(format!("bv_atac_frag_{tag}_{}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempPath(p)
    }

    #[test]
    fn test_parse_u32() {
        assert_eq!(parse_u32(b"0"), Some(0));
        assert_eq!(parse_u32(b"4294967295"), Some(u32::MAX));
        assert_eq!(parse_u32(b"4294967296"), None);
        assert_eq!(parse_u32(b""), None);
        assert_eq!(parse_u32(b"12a"), None);
    }

    #[test]
    fn test_split_edges() {
        assert_eq!(
            split_edges(b"ab\ncd\nef", false),
            (&b"ab\n"[..], &b"cd\n"[..], &b"ef"[..])
        );
        assert_eq!(
            split_edges(b"ab\ncd\n", true),
            (&b""[..], &b"ab\ncd\n"[..], &b""[..])
        );
        assert_eq!(
            split_edges(b"abc", false),
            (&b"abc"[..], &b""[..], &b""[..])
        );
    }

    #[test]
    fn test_all_compressions_read_the_same_fragments() {
        let text = synthetic_text(3000);
        let cells: Vec<String> = ["CCCG-1", "TTTA-1", "AAAC-1"].map(String::from).to_vec();

        let plain = temp_file("plain", text.as_bytes());
        let mut gz = GzEncoder::new(Vec::new(), GzLevel::fast());
        gz.write_all(text.as_bytes()).unwrap();
        let gzip = temp_file("gzip", &gz.finish().unwrap());
        // small blocks so lines straddle block and group boundaries
        let bgzf = temp_file("bgzf", &bgzf_bytes(text.as_bytes(), 997));

        let reference = collect(&plain.0, &cells);
        assert_eq!(reference.len(), 3 * 3000 * 3 / 4);
        assert_eq!(collect(&gzip.0, &cells), reference);
        assert_eq!(collect(&bgzf.0, &cells), reference);
        assert!(reference.iter().all(|(_, f)| f.cell < 3 && f.count >= 1));
    }

    #[test]
    fn test_malformed_line_is_an_error() {
        let f = temp_file("bad", b"chr1\t10\t5\tAAAC-1\t1\n");
        let cells = vec!["AAAC-1".to_string()];
        let idx = BarcodeIndex::new(&cells).unwrap();
        let r = fold_fragments(&f.0, &idx, || 0usize, |a, _, f| *a += f.len(), |a, b| a + b);
        assert!(r.is_err());
    }

    /// A minimal tabix index for text written by [`bgzf_bytes`]: one bin and
    /// one chunk per chromosome spanning its lines.
    ///
    /// ### Params
    ///
    /// * `text` - The file text
    /// * `bgzf` - Its BGZF bytes
    /// * `block` - Text bytes per block used for `bgzf`
    ///
    /// ### Returns
    ///
    /// The gzip compressed index.
    fn tabix_bytes(text: &str, bgzf: &[u8], block: usize) -> Vec<u8> {
        let coff = bgzf_block_offsets(bgzf).unwrap();
        let voff = |u: usize| ((coff[u / block] as u64) << 16) | (u % block) as u64;
        let mut spans: Vec<(String, usize, usize)> = Vec::new();
        let mut pos = 0;
        for line in text.split_inclusive('\n') {
            if !line.starts_with('#') {
                let chrom = line.split('\t').next().unwrap().to_string();
                match spans.last_mut() {
                    Some(s) if s.0 == chrom => s.2 = pos + line.len(),
                    _ => spans.push((chrom, pos, pos + line.len())),
                }
            }
            pos += line.len();
        }
        let mut idx: Vec<u8> = b"TBI\x01".to_vec();
        let i32s = |v: &mut Vec<u8>, xs: &[i32]| xs.iter().for_each(|x| v.extend(x.to_le_bytes()));
        let names: Vec<u8> = spans
            .iter()
            .flat_map(|s| s.0.bytes().chain(std::iter::once(0)))
            .collect();
        i32s(
            &mut idx,
            &[spans.len() as i32, 0, 1, 2, 3, 35, 0, names.len() as i32],
        );
        idx.extend(&names);
        for (_, b, e) in &spans {
            i32s(&mut idx, &[1, 4681, 1]);
            idx.extend(voff(*b).to_le_bytes());
            idx.extend(voff(*e).to_le_bytes());
            i32s(&mut idx, &[0]);
        }
        let mut gz = GzEncoder::new(Vec::new(), GzLevel::fast());
        gz.write_all(&idx).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn test_span_folds_read_exactly_their_chromosomes() {
        let text = synthetic_text(3000);
        let cells: Vec<String> = ["CCCG-1", "TTTA-1", "AAAC-1"].map(String::from).to_vec();
        // small blocks so chromosome boundaries fall inside blocks
        let bgzf = bgzf_bytes(text.as_bytes(), 997);
        let file = temp_file("spans", &bgzf);
        let mut tbi = file.0.clone().into_os_string();
        tbi.push(".tbi");
        let tbi = TempPath(tbi.into());
        std::fs::write(&tbi.0, tabix_bytes(&text, &bgzf, 997)).unwrap();

        let spans = chrom_spans(&file.0).unwrap().unwrap();
        let names: Vec<&str> = spans.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["chr1", "chr2", "chrX"]);

        let all = collect(&file.0, &cells);
        let idx = BarcodeIndex::new(&cells).unwrap();
        for pick in [vec![1], vec![0, 2], vec![0, 1, 2]] {
            let chosen: Vec<ChromSpan> = pick.iter().map(|&i| spans[i].clone()).collect();
            let mut got = fold_fragment_spans(
                &file.0,
                &chosen,
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
            got.sort_by(|a, b| (&a.0, a.1.start, a.1.cell).cmp(&(&b.0, b.1.start, b.1.cell)));
            let want: Vec<_> = all
                .iter()
                .filter(|(c, _)| chosen.iter().any(|s| &s.name == c))
                .cloned()
                .collect();
            assert_eq!(got, want, "spans {pick:?}");
        }
    }

    #[test]
    fn test_no_index_means_no_spans() {
        let f = temp_file("noidx", &bgzf_bytes(synthetic_text(10).as_bytes(), 997));
        assert!(chrom_spans(&f.0).unwrap().is_none());
    }

    #[test]
    fn test_duplicate_barcodes_are_refused() {
        let cells: Vec<String> = ["A", "A"].map(String::from).to_vec();
        assert!(BarcodeIndex::new(&cells).is_err());
    }
}
