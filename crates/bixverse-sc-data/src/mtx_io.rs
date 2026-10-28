//! Matrix Market (.mtx) ingestion into the bixverse cell store.
//!
//! The text is parsed once. Plain files are cut into newline-aligned blocks
//! read with positional reads; gzip input (found by its magic bytes,
//! multi-member included) cannot be split, so one thread inflates it and
//! feeds blocks through a bounded channel to the parallel parsers. That pass
//! counts gene nnz, totals every cell and spills each entry into binary bucket
//! files keyed by the original cell. The cell QC is resolved from those totals
//! without a second pass (see [`TextPassAcc`]), and the write reads the
//! buckets back. Input already in cell order skips the per-bucket sort.
//!
//! Values are raw counts. `integer` and `real` fields must hold non-negative
//! integral values (a real `1.5` is an error, not a silent `1`); `pattern`
//! entries count 1. `complex` fields and any symmetry other than `general`
//! are rejected.

use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::Instant;
use thousands::Separable;

use crate::data_io::{
    CellGeneSparseWriter, CellOnFileQuality, INDEX_DROPPED, compress_cell_row, dense_index_map,
};
use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Byte width of one record in a temporary bucket spill file:
/// `cell: u32`, `gene: u32`, `count: u32`, original 0-based indices.
///
/// These files live and die inside one [`MtxReader`], so the layout carries
/// no on-disk compatibility obligation.
const BUCKET_RECORD_LEN: usize = 12;

/// Bytes of mtx text per block, the unit of parallel parsing. Every worker
/// holds one block, and the gzip channel holds up to
/// [`GZ_BLOCKS_PER_THREAD`] per worker, so this times the thread count bounds
/// the text in flight.
const MTX_BLOCK_BYTES: usize = 8 * 1024 * 1024;

/// Blocks the gzip reader may run ahead of the parsers, per worker. Enough to
/// keep the parsers fed across the jitter of a single inflate thread.
const GZ_BLOCKS_PER_THREAD: usize = 2;

/// Size at which a worker's per-bucket record buffer is appended to the
/// shared bucket file. Caps the buffers at
/// `n_threads * n_buckets * BUCKET_FLUSH_BYTES`.
const BUCKET_FLUSH_BYTES: usize = 64 * 1024;

/// Most bucket files open at once, well below the macOS fd limit.
const MAX_BUCKETS: usize = 128;

/// Original cells per bucket before [`MAX_BUCKETS`] caps the count. Keeps a
/// bucket, read back whole in the write pass, small next to the matrix.
const CELLS_PER_BUCKET: usize = 5_000;

/// Relative tolerance for a `real` value to count as integral. Writers that
/// print floats round-trip an integer exactly; anything further off is not a
/// raw count.
const REAL_INTEGRAL_TOL: f64 = 1e-6;

/// The two magic bytes that open every gzip member.
const GZ_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Distinguishes the spill files of concurrent readers in one process.
static SPILL_COUNTER: AtomicUsize = AtomicUsize::new(0);

////////////
// Header //
////////////

/// Value type of an mtx file, from the `%%MatrixMarket` banner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtxField {
    /// Integer values, also assumed when the banner is missing.
    Integer,
    /// Floating point values; each must be a non-negative integral count.
    Real,
    /// No value column; every entry counts 1.
    Pattern,
}

/// MTX file metadata
#[derive(Debug, Clone)]
pub struct MtxHeader {
    /// Number of cells identified in the .mtx header.
    pub total_cells: usize,
    /// Number of genes identified in the .mtx header.
    pub total_genes: usize,
    /// Number of entries identified in the .mtx header.
    pub total_entries: usize,
    /// Value type from the banner.
    pub field: MtxField,
}

/// Parse the `%%MatrixMarket` banner.
///
/// ### Params
///
/// * `line` - The banner line, newline included or not
///
/// ### Returns
///
/// The value type, or an error for anything other than a general coordinate
/// matrix of integer, real or pattern values.
fn parse_banner(line: &[u8]) -> Result<MtxField, ScDataErrors> {
    let text = String::from_utf8_lossy(line).to_ascii_lowercase();
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let [_, object, format, field, symmetry] = tokens[..] else {
        return Err(ScDataErrors::MtxHeaderInvalid(
            "the banner needs object, format, field and symmetry",
        ));
    };
    if object != "matrix" {
        return Err(ScDataErrors::MtxUnsupported(format!("object '{object}'")));
    }
    if format != "coordinate" {
        return Err(ScDataErrors::MtxUnsupported(format!(
            "format '{format}', only sparse 'coordinate' is read"
        )));
    }
    if symmetry != "general" {
        return Err(ScDataErrors::MtxUnsupported(format!(
            "symmetry '{symmetry}', only 'general' is read"
        )));
    }
    match field {
        "integer" => Ok(MtxField::Integer),
        "real" | "double" => Ok(MtxField::Real),
        "pattern" => Ok(MtxField::Pattern),
        other => Err(ScDataErrors::MtxUnsupported(format!(
            "field '{other}', counts must be integer, real or pattern"
        ))),
    }
}

/// Read the banner, comments and shape line.
///
/// ### Params
///
/// * `reader` - Stream positioned at the start of the (decompressed) file
/// * `cells_as_rows` - Orientation, names the failing field in errors
///
/// ### Returns
///
/// `(field, [rows, cols, entries], bytes consumed)`.
fn parse_preamble(
    reader: &mut impl BufRead,
    cells_as_rows: bool,
) -> Result<(MtxField, [usize; 3], u64), ScDataErrors> {
    let mut line = Vec::new();
    let mut consumed = 0u64;
    let mut field = MtxField::Integer;
    let mut first = true;
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 {
            return Err(ScDataErrors::MtxHeaderInvalid("no shape line"));
        }
        consumed += n as u64;
        if first && line.starts_with(b"%%") {
            field = parse_banner(&line)?;
        }
        first = false;
        if line.first() != Some(&b'%') {
            break;
        }
    }

    let text = std::str::from_utf8(&line)
        .map_err(|_| ScDataErrors::MtxHeaderInvalid("the shape line is not text"))?;
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(ScDataErrors::MtxHeaderInvalid(
            "expected three whitespace-separated fields on the shape line",
        ));
    }
    let names = if cells_as_rows {
        ["cell count", "gene count", "entry count"]
    } else {
        ["gene count", "cell count", "entry count"]
    };
    let mut shape = [0usize; 3];
    for (k, (s, field)) in parts.iter().zip(names).enumerate() {
        shape[k] = s
            .parse()
            .map_err(|_| ScDataErrors::MtxParseError { field })?;
    }
    Ok((field, shape, consumed))
}

////////////
// Source //
////////////

/// An mtx file, plain or gzipped, read as blocks of whole lines.
pub(crate) struct MtxSource {
    /// Path to the file
    path: PathBuf,
    /// Whether the file is gzip, by magic bytes
    gz: bool,
    /// Value type from the banner
    pub(crate) field: MtxField,
    /// `[rows, cols, entries]` from the shape line
    pub(crate) shape: [usize; 3],
    /// Offset of the first entry line in the (decompressed) stream
    data_start: u64,
    /// Target block size, [`MTX_BLOCK_BYTES`] outside of tests
    block_bytes: usize,
}

impl MtxSource {
    /// Open an mtx file and read its header.
    ///
    /// ### Params
    ///
    /// * `path` - Plain or gzipped mtx
    /// * `cells_as_rows` - Orientation, names the failing field in errors
    ///
    /// ### Returns
    ///
    /// The source, ready to be read block by block.
    pub(crate) fn open(path: &Path, cells_as_rows: bool) -> Result<Self, ScDataErrors> {
        let mut magic = [0u8; 2];
        let mut file = File::open(path)?;
        let gz = file.read(&mut magic)? == 2 && magic == GZ_MAGIC;
        file.rewind()?;
        let mut source = Self {
            path: path.to_path_buf(),
            gz,
            field: MtxField::Integer,
            shape: [0; 3],
            data_start: 0,
            block_bytes: MTX_BLOCK_BYTES,
        };
        let mut reader = BufReader::new(source.stream()?);
        let (field, shape, data_start) = parse_preamble(&mut reader, cells_as_rows)?;
        source.field = field;
        source.shape = shape;
        source.data_start = data_start;
        Ok(source)
    }

    /// The file as a byte stream from its start, inflated if gzipped.
    ///
    /// ### Returns
    ///
    /// A reader over the (decompressed) text.
    fn stream(&self) -> Result<Box<dyn Read + Send>, ScDataErrors> {
        let file = File::open(&self.path)?;
        Ok(if self.gz {
            Box::new(MultiGzDecoder::new(file))
        } else {
            Box::new(file)
        })
    }

    /// Hand every block of whole entry lines to `f`, in file order, on the
    /// calling thread.
    ///
    /// ### Params
    ///
    /// * `f` - Called with the block's sequence number and its bytes
    ///
    /// ### Returns
    ///
    /// The first error of the read or of `f`.
    pub(crate) fn for_each_block_seq(
        &self,
        mut f: impl FnMut(usize, Vec<u8>) -> Result<(), ScDataErrors>,
    ) -> Result<(), ScDataErrors> {
        let mut reader = self.stream()?;
        std::io::copy(
            &mut (&mut reader).take(self.data_start),
            &mut std::io::sink(),
        )?;

        let mut carry = Vec::new();
        let mut seq = 0;
        loop {
            let mut block = std::mem::take(&mut carry);
            block.reserve(self.block_bytes);
            let n = (&mut reader)
                .take(self.block_bytes as u64)
                .read_to_end(&mut block)?;
            if n == 0 {
                if !block.is_empty() {
                    f(seq, block)?;
                }
                return Ok(());
            }
            // a line longer than a block stays in the carry until it ends
            if let Some(p) = block.iter().rposition(|&b| b == b'\n') {
                carry = block[p + 1..].to_vec();
                block.truncate(p + 1);
                f(seq, block)?;
                seq += 1;
            } else {
                carry = block;
            }
        }
    }

    /// Newline-aligned byte ranges of a plain file's entry lines.
    ///
    /// ### Returns
    ///
    /// Half-open ranges of about `block_bytes`, covering every line once.
    fn plain_ranges(&self) -> Result<Vec<(u64, u64)>, ScDataErrors> {
        let mut file = File::open(&self.path)?;
        let len = file.metadata()?.len();
        let mut ranges = Vec::new();
        let mut a = self.data_start;
        let mut probe = vec![0u8; 64 * 1024];
        while a < len {
            let mut b = a + self.block_bytes as u64;
            if b >= len {
                ranges.push((a, len));
                break;
            }
            file.seek(SeekFrom::Start(b))?;
            loop {
                let k = file.read(&mut probe)?;
                if k == 0 {
                    b = len;
                    break;
                }
                if let Some(p) = probe[..k].iter().position(|&c| c == b'\n') {
                    b += p as u64 + 1;
                    break;
                }
                b += k as u64;
            }
            ranges.push((a, b));
            a = b;
        }
        Ok(ranges)
    }

    /// Fold every block in parallel into one accumulator per worker thread.
    ///
    /// Plain files are read as positional block reads by the workers. Gzip is
    /// inflated on one extra thread that sends blocks through a bounded
    /// channel. Block order is not preserved; `f` gets each block's sequence
    /// number to restore it.
    ///
    /// ### Params
    ///
    /// * `init` - Builds a worker's accumulator on its first block
    /// * `f` - Folds one block into the worker's accumulator
    ///
    /// ### Returns
    ///
    /// The accumulators of the workers that saw a block.
    pub(crate) fn par_fold_blocks<A, I, F>(&self, init: I, f: F) -> Result<Vec<A>, ScDataErrors>
    where
        A: Send,
        I: Fn() -> A + Sync,
        F: Fn(&mut A, usize, &[u8]) -> Result<(), ScDataErrors> + Sync,
    {
        // One slot per worker. A worker never runs rayon work inside `f`, so
        // it never waits on its own slot.
        let slots: Vec<Mutex<Option<A>>> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(None))
            .collect();
        let fold = |seq: usize, bytes: &[u8]| -> Result<(), ScDataErrors> {
            let k = rayon::current_thread_index().unwrap_or(0) % slots.len();
            let mut slot = slots[k]
                .lock()
                .expect("an accumulator lock is never held across a panic");
            f(slot.get_or_insert_with(&init), seq, bytes)
        };

        if self.gz {
            let (tx, rx) = sync_channel::<(usize, Vec<u8>)>(slots.len() * GZ_BLOCKS_PER_THREAD);
            std::thread::scope(|s| {
                let reader = s.spawn(move || {
                    self.for_each_block_seq(|seq, block| {
                        // a closed channel means the parsers stopped on an
                        // error, which they report
                        tx.send((seq, block)).map_err(|_| {
                            ScDataErrors::BinaryIo(std::io::Error::other("mtx parsers stopped"))
                        })
                    })
                });
                let parsed = rx
                    .into_iter()
                    .par_bridge()
                    .try_for_each(|(seq, block)| fold(seq, &block));
                let read = reader.join().expect("the gzip reader thread panicked");
                parsed.and(read)
            })?;
        } else {
            self.plain_ranges()?
                .par_iter()
                .enumerate()
                .try_for_each_init(
                    Vec::new,
                    |buf, (seq, &(a, b))| -> Result<(), ScDataErrors> {
                        let mut file = File::open(&self.path)?;
                        file.seek(SeekFrom::Start(a))?;
                        buf.resize((b - a) as usize, 0);
                        file.read_exact(buf)?;
                        fold(seq, buf)
                    },
                )?;
        }

        Ok(slots
            .into_iter()
            .filter_map(|m| {
                m.into_inner()
                    .expect("an accumulator lock is never held across a panic")
            })
            .collect())
    }
}

/////////////
// Parsing //
/////////////

/// Parse an unsigned decimal at `line[*i..]`, advancing `i`.
///
/// ### Params
///
/// * `line` - Line bytes
/// * `i` - Cursor
///
/// ### Returns
///
/// The value (wrapping past `u32::MAX`, which no mtx index reaches), `None`
/// if no digit is at the cursor.
#[inline]
fn parse_index(line: &[u8], i: &mut usize) -> Option<u32> {
    let start = *i;
    let mut v = 0u32;
    while *i < line.len() && line[*i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((line[*i] - b'0') as u32);
        *i += 1;
    }
    (*i > start).then_some(v)
}

/// Advance `i` past spaces and tabs.
///
/// ### Params
///
/// * `line` - Line bytes
/// * `i` - Cursor
#[inline]
fn skip_blank(line: &[u8], i: &mut usize) {
    while *i < line.len() && (line[*i] == b' ' || line[*i] == b'\t') {
        *i += 1;
    }
}

/// Parse a value token as a raw count.
///
/// Plain digits take the fast path. Anything else (`2.0`, `1.5e+01`, `-1`)
/// is read as a float and must be finite, non-negative and integral within
/// [`REAL_INTEGRAL_TOL`].
///
/// ### Params
///
/// * `token` - The value field, no surrounding blanks
///
/// ### Returns
///
/// The count, or [`ScDataErrors::MtxNotACount`].
#[inline]
fn parse_count(token: &[u8]) -> Result<u32, ScDataErrors> {
    let not_a_count = || ScDataErrors::MtxNotACount {
        value: String::from_utf8_lossy(token).into_owned(),
    };
    if !token.is_empty() && token.iter().all(u8::is_ascii_digit) {
        let mut v = 0u64;
        for &b in token {
            v = v * 10 + (b - b'0') as u64;
            if v > u32::MAX as u64 {
                return Err(not_a_count());
            }
        }
        return Ok(v as u32);
    }
    let v: f64 = std::str::from_utf8(token)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(not_a_count)?;
    let r = v.round();
    if !v.is_finite()
        || r < 0.0
        || r > u32::MAX as f64
        || (v - r).abs() > REAL_INTEGRAL_TOL * r.abs().max(1.0)
    {
        return Err(not_a_count());
    }
    Ok(r as u32)
}

/// Parse one entry line.
///
/// ### Params
///
/// * `line` - The line without its terminator
/// * `field` - Value type
///
/// ### Returns
///
/// `Some((row, col, count))` with 1-based coordinates as stored, `None` for a
/// line that is not an entry (blank, comment, too few fields), an error for a
/// value that is not a raw count.
#[inline]
fn parse_entry(line: &[u8], field: MtxField) -> Result<Option<(u32, u32, u32)>, ScDataErrors> {
    let mut i = 0;
    let Some(row) = parse_index(line, &mut i) else {
        return Ok(None);
    };
    skip_blank(line, &mut i);
    let Some(col) = parse_index(line, &mut i) else {
        return Ok(None);
    };
    if field == MtxField::Pattern {
        return Ok(Some((row, col, 1)));
    }
    skip_blank(line, &mut i);
    let start = i;
    while i < line.len() && !matches!(line[i], b' ' | b'\t') {
        i += 1;
    }
    if i == start {
        return Ok(None);
    }
    Ok(Some((row, col, parse_count(&line[start..i])?)))
}

/// Parse every entry line of a block.
///
/// ### Params
///
/// * `bytes` - Whole lines
/// * `field` - Value type
/// * `f` - Called with `(row, col, count)` per entry, 1-based as stored
///
/// ### Returns
///
/// The first value error.
pub(crate) fn parse_block<F: FnMut(u32, u32, u32)>(
    bytes: &[u8],
    field: MtxField,
    mut f: F,
) -> Result<(), ScDataErrors> {
    for line in bytes.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some((row, col, value)) = parse_entry(line, field)? {
            f(row, col, value);
        }
    }
    Ok(())
}

///////////
// Spill //
///////////

/// RAII guard that removes temporary bucket files on drop.
///
/// Ensures temp files are cleaned up even if a pass returns early due to an
/// I/O error or panic.
struct TempFileGuard(Vec<PathBuf>);

/// Drop implementation for `TempFileGuard`
///
/// Attempts to remove each tracked temp file. Errors are silently ignored since
/// this runs during unwind and the files may already have been removed by the
/// writing pass.
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// One bucket file while the text pass appends to it.
struct BucketFile {
    /// Buffered handle
    writer: BufWriter<File>,
    /// Bytes appended so far
    len: u64,
    /// `(block seq, offset, length)` of every appended segment
    segments: Vec<(usize, u64, u64)>,
}

/// Cell order seen in one block.
struct BlockOrder {
    /// Block sequence number
    seq: usize,
    /// First cell of the block
    first: u32,
    /// Last cell of the block
    last: u32,
    /// Whether cells never decreased inside the block
    sorted: bool,
}

/// Per-worker state of the text pass.
///
/// QC without a second pass: a gene is dropped when it is seen in fewer than
/// `min_cells` cells, so while a worker has seen a gene fewer than
/// `min_cells` times it keeps that gene's `(cell, count)` entries. Once the
/// count is reached the gene is certainly kept and the list goes. After the
/// merge, the entries of the dropped genes are exactly the union of the lists
/// still held, and subtracting them from the all-gene cell totals gives the
/// totals over kept genes.
struct TextPassAcc {
    /// Entries per gene
    gene_nnz: Vec<u32>,
    /// `(cell, count)` per gene while the gene is below `min_cells` here
    rare: Vec<Vec<(u32, u32)>>,
    /// `(nnz, library size)` per cell over all genes
    cell_stats: Vec<(u32, u32)>,
    /// Record buffer per bucket
    bufs: Vec<Vec<u8>>,
    /// Cell order of every block this worker parsed
    blocks: Vec<BlockOrder>,
}

/// Entries spilled by the text pass, and the QC it resolved.
struct Spill {
    /// Bucket files, removed on drop
    files: TempFileGuard,
    /// Original cells per bucket
    cells_per_bucket: usize,
    /// `(block seq, offset, length)` of every segment, per bucket
    segments: Vec<Vec<(usize, u64, u64)>>,
    /// Whether the file listed cells in non-decreasing order
    cell_major: bool,
    /// QC from the text pass
    quality: CellOnFileQuality,
}

////////////
// Reader //
////////////

/// MTX final data
///
/// Structure to store final results after reading in the .mtx file
#[derive(Debug, Clone)]
pub struct MtxFinalData {
    /// Structure containing the information on which cells/genes to keep and
    /// library size and NNZ for cells.
    pub cell_qc: CellQuality,
    /// No of genes that were read in
    pub no_genes: usize,
    /// No of cells that were read in
    pub no_cells: usize,
}

/// MTX Reader for bixverse
///
/// Parses the text once, in [`Self::parse_mtx_quality`] or on demand in the
/// write, and keeps the spilled entries between the two calls, so use one
/// reader for both.
pub struct MtxReader {
    /// The input file
    source: MtxSource,
    /// Header of the MtxFile
    header: MtxHeader,
    /// Cell and gene quality parameters to apply
    qc_params: MinCellQuality,
    /// Are the cells stored as rows
    cells_as_rows: bool,
    /// Where the bucket files go
    temp_dir: PathBuf,
    /// Spilled entries, after the text pass
    spill: Option<Spill>,
}

impl MtxReader {
    /// Generate a new instance of the reader
    ///
    /// The file may be plain or gzipped (detected by its magic bytes). Bucket
    /// files go to the system temp directory unless [`Self::with_temp_dir`]
    /// says otherwise; they take 12 bytes per entry.
    ///
    /// ### Params
    ///
    /// * `path` - Path to the mtx file.
    /// * `qc_params` - The min quality parameters that genes and cells have to
    ///   reach and the target library size
    /// * `cells_as_rows` - Boolean. Are the cells the rows (= true) or columns in
    ///   the mtx file.
    pub fn new<P: AsRef<Path>>(
        path: P,
        qc_params: MinCellQuality,
        cells_as_rows: bool,
    ) -> Result<Self, ScDataErrors> {
        let source = MtxSource::open(path.as_ref(), cells_as_rows)?;
        let [rows, cols, entries] = source.shape;
        let (total_cells, total_genes) = if cells_as_rows {
            (rows, cols)
        } else {
            (cols, rows)
        };
        let header = MtxHeader {
            total_cells,
            total_genes,
            total_entries: entries,
            field: source.field,
        };
        Ok(Self {
            source,
            header,
            qc_params,
            cells_as_rows,
            temp_dir: std::env::temp_dir(),
            spill: None,
        })
    }

    /// Put the bucket files in `dir` instead of the system temp directory.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory for the spill, 12 bytes per entry
    ///
    /// ### Returns
    ///
    /// The reader.
    pub fn with_temp_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.temp_dir = dir.as_ref().to_path_buf();
        self
    }

    /// Parse the file and resolve which cells and genes pass QC.
    ///
    /// The single text pass: gene nnz for `min_cells`, then per-cell nnz and
    /// library size over the kept genes for `min_unique_genes` and
    /// `min_lib_size`. The parsed entries stay spilled on disk for the write.
    ///
    /// ### Params
    ///
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The CellOnFileQuality file containing the indices and mappings
    /// for the cells/genes to keep.
    pub fn parse_mtx_quality(&mut self, verbose: bool) -> Result<CellOnFileQuality, ScDataErrors> {
        if self.spill.is_none() {
            self.spill = Some(self.text_pass(verbose)?);
        }
        Ok(self
            .spill
            .as_ref()
            .expect("spill set above")
            .quality
            .clone())
    }

    /// The one text pass: parse, count, spill, resolve QC.
    ///
    /// ### Params
    ///
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// The spill with its QC.
    fn text_pass(&self, verbose: bool) -> Result<Spill, ScDataErrors> {
        let start = Instant::now();
        let n_cells = self.header.total_cells;
        let n_genes = self.header.total_genes;
        let min_cells = self.qc_params.min_cells;
        let cells_as_rows = self.cells_as_rows;
        let field = self.source.field;

        let n_buckets = n_cells.div_ceil(CELLS_PER_BUCKET).clamp(1, MAX_BUCKETS);
        let cells_per_bucket = n_cells.div_ceil(n_buckets).max(1);

        let tag = SPILL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let paths: Vec<PathBuf> = (0..n_buckets)
            .map(|i| {
                self.temp_dir.join(format!(
                    ".bixverse_mtx_{}_{tag}.bucket_{i:04}.tmp",
                    std::process::id()
                ))
            })
            .collect();
        let files = TempFileGuard(paths.clone());
        let buckets: Vec<Mutex<BucketFile>> = paths
            .iter()
            .map(|p| {
                File::create(p).map(|f| {
                    Mutex::new(BucketFile {
                        writer: BufWriter::with_capacity(256 * 1024, f),
                        len: 0,
                        segments: Vec::new(),
                    })
                })
            })
            .collect::<std::io::Result<_>>()?;

        if verbose {
            let kind = match field {
                MtxField::Integer => "integer",
                MtxField::Real => "real",
                MtxField::Pattern => "pattern",
            };
            println!(
                "Text pass ({}{kind}): {n_buckets} buckets, ~{cells_per_bucket} cells/bucket",
                if self.source.gz { "gzip, " } else { "" },
            );
        }

        let flush = |bucket: usize, seq: usize, buf: &mut Vec<u8>| -> Result<(), ScDataErrors> {
            let mut b = buckets[bucket]
                .lock()
                .expect("bucket writer lock is never held across a panic");
            let len = buf.len() as u64;
            let offset = b.len;
            b.writer.write_all(buf)?;
            b.segments.push((seq, offset, len));
            b.len += len;
            buf.clear();
            Ok(())
        };

        let accs = self.source.par_fold_blocks(
            || TextPassAcc {
                gene_nnz: vec![0; n_genes],
                rare: vec![Vec::new(); n_genes],
                cell_stats: vec![(0, 0); n_cells],
                bufs: vec![Vec::new(); n_buckets],
                blocks: Vec::new(),
            },
            |acc, seq, bytes| {
                let mut order: Option<BlockOrder> = None;
                let mut flush_err = Ok(());
                parse_block(bytes, field, |row, col, value| {
                    let (cell, gene) = if cells_as_rows {
                        (row, col)
                    } else {
                        (col, row)
                    };
                    let Some(g) = (gene as usize).checked_sub(1).filter(|&g| g < n_genes) else {
                        return;
                    };
                    acc.gene_nnz[g] += 1;
                    let Some(c) = (cell as usize).checked_sub(1).filter(|&c| c < n_cells) else {
                        return;
                    };
                    let seen = acc.gene_nnz[g] as usize;
                    if seen < min_cells {
                        acc.rare[g].push((c as u32, value));
                    } else if seen == min_cells {
                        acc.rare[g] = Vec::new();
                    }
                    let stats = &mut acc.cell_stats[c];
                    stats.0 += 1;
                    stats.1 += value;

                    let c32 = c as u32;
                    match &mut order {
                        None => {
                            order = Some(BlockOrder {
                                seq,
                                first: c32,
                                last: c32,
                                sorted: true,
                            })
                        }
                        Some(o) => {
                            o.sorted &= c32 >= o.last;
                            o.last = c32;
                        }
                    }

                    let bucket = c / cells_per_bucket;
                    let buf = &mut acc.bufs[bucket];
                    buf.extend_from_slice(&c32.to_le_bytes());
                    buf.extend_from_slice(&(g as u32).to_le_bytes());
                    buf.extend_from_slice(&value.to_le_bytes());
                    if buf.len() >= BUCKET_FLUSH_BYTES && flush_err.is_ok() {
                        flush_err = flush(bucket, seq, buf);
                    }
                })?;
                flush_err?;
                // every segment must belong to one block, so the cell order
                // can be restored from the block sequence
                for (bucket, buf) in acc.bufs.iter_mut().enumerate() {
                    if !buf.is_empty() {
                        flush(bucket, seq, buf)?;
                    }
                }
                acc.blocks.extend(order);
                Ok(())
            },
        )?;

        let mut segments = Vec::with_capacity(n_buckets);
        for bucket in buckets {
            let mut b = bucket
                .into_inner()
                .expect("bucket writer lock is never held across a panic");
            b.writer.flush()?;
            segments.push(b.segments);
        }

        // merge the workers
        let mut gene_nnz = vec![0usize; n_genes];
        let mut cell_stats = vec![(0u32, 0u32); n_cells];
        let mut blocks = Vec::new();
        for acc in &accs {
            gene_nnz
                .iter_mut()
                .zip(&acc.gene_nnz)
                .for_each(|(t, &n)| *t += n as usize);
            cell_stats
                .par_iter_mut()
                .zip(&acc.cell_stats)
                .for_each(|(t, s)| {
                    t.0 += s.0;
                    t.1 += s.1;
                });
        }
        let genes_to_keep: Vec<usize> =
            (0..n_genes).filter(|&g| gene_nnz[g] >= min_cells).collect();
        for mut acc in accs {
            for (g, rare) in acc.rare.iter().enumerate() {
                if gene_nnz[g] < min_cells {
                    for &(c, v) in rare {
                        let s = &mut cell_stats[c as usize];
                        s.0 -= 1;
                        s.1 -= v;
                    }
                }
            }
            blocks.append(&mut acc.blocks);
        }

        blocks.sort_unstable_by_key(|b| b.seq);
        let cell_major =
            blocks.iter().all(|b| b.sorted) && blocks.windows(2).all(|w| w[0].last <= w[1].first);

        let cells_to_keep: Vec<usize> = (0..n_cells)
            .filter(|&c| {
                cell_stats[c].0 as usize >= self.qc_params.min_unique_genes
                    && cell_stats[c].1 as f32 >= self.qc_params.min_lib_size as f32
            })
            .collect();

        let mut quality = CellOnFileQuality::new(cells_to_keep, genes_to_keep);
        quality.generate_maps_sets();

        if verbose {
            println!(
                "Text pass done: {:.2?} ({} order)",
                start.elapsed(),
                if cell_major { "cell-major" } else { "unsorted" }
            );
            println!(
                "Genes passing QC: {} / {}",
                quality.genes_to_keep.len().separate_with_underscores(),
                n_genes.separate_with_underscores()
            );
            println!(
                "Cells passing QC: {} / {}",
                quality.cells_to_keep.len().separate_with_underscores(),
                n_cells.separate_with_underscores()
            );
        }

        Ok(Spill {
            files,
            cells_per_bucket,
            segments,
            cell_major,
            quality,
        })
    }

    /// Process the mtx file and write to binarised Rust file
    ///
    /// Calls [`Self::process_mtx_and_write_bin_streaming`]; the bucketed write
    /// is the only one left, so both entry points give the same store.
    ///
    /// ### Params
    ///
    /// * `bin_path` - Where to save the binarised file.
    /// * `quality` - Structure indicating which cells and genes to keep from
    ///   the mtx file.
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The `MtxFinalData` with information how many cells were written to
    /// file, how many genes were included in which cells did not parse
    /// the thresholds.
    pub fn process_mtx_and_write_bin(
        self,
        bin_path: &str,
        quality: &CellOnFileQuality,
        verbose: bool,
    ) -> Result<MtxFinalData, ScDataErrors> {
        self.process_mtx_and_write_bin_streaming(bin_path, quality, verbose)
    }

    /// Process the mtx file and write to binarised Rust file
    ///
    /// Reads the bucket files of the text pass (running it first if
    /// [`Self::parse_mtx_quality`] was not called) one at a time, so memory is
    /// bounded by a bucket. A file listed in cell order needs no sort; any
    /// other order sorts each bucket on the full `(cell, gene, count)` tuple,
    /// which makes the output deterministic. Every kept cell is written,
    /// empty or not.
    ///
    /// ### Params
    ///
    /// * `bin_path` - Where to save the binarised file.
    /// * `quality` - Structure indicating which cells and genes to keep from
    ///   the mtx file. `cells_to_keep` must be ascending.
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// The `MtxFinalData` with information how many cells were written to
    /// file, how many genes were included and the per-cell library size and
    /// NNZ statistics.
    pub fn process_mtx_and_write_bin_streaming(
        mut self,
        bin_path: &str,
        quality: &CellOnFileQuality,
        verbose: bool,
    ) -> Result<MtxFinalData, ScDataErrors> {
        let kept = &quality.cells_to_keep;
        if kept.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ScDataErrors::InvalidArgument(
                "cells_to_keep must be strictly ascending".into(),
            ));
        }
        let n_kept_cells = kept.len();
        let n_kept_genes = quality.genes_to_keep.len();
        let target_size = self.qc_params.target_size;

        if n_kept_cells == 0 {
            let writer = CellGeneSparseWriter::new(bin_path, true, 0, n_kept_genes, target_size)?;
            writer.finalise()?;
            return Ok(MtxFinalData {
                cell_qc: CellQuality {
                    cell_indices: vec![],
                    gene_indices: quality.genes_to_keep.to_vec(),
                    lib_size: vec![],
                    nnz: vec![],
                },
                no_genes: n_kept_genes,
                no_cells: 0,
            });
        }

        let spill = match self.spill.take() {
            Some(s) => s,
            None => self.text_pass(verbose)?,
        };

        let start = Instant::now();
        if verbose {
            println!(
                "Writing pass: {} buckets{}",
                spill.files.0.len(),
                if spill.cell_major { ", no sort" } else { "" }
            );
        }

        let cell_map = dense_index_map(kept);
        let gene_map = dense_index_map(&quality.genes_to_keep);
        let lookup = |map: &[u32], i: u32| -> Option<u32> {
            map.get(i as usize).copied().filter(|&m| m != INDEX_DROPPED)
        };

        let mut writer =
            CellGeneSparseWriter::new(bin_path, true, n_kept_cells, n_kept_genes, target_size)?;
        let mut lib_size = Vec::with_capacity(n_kept_cells);
        let mut nnz = Vec::with_capacity(n_kept_cells);
        let n_buckets = spill.files.0.len();

        for (bucket_idx, temp_path) in spill.files.0.iter().enumerate() {
            let raw = std::fs::read(temp_path)?;
            let bytes = if spill.cell_major {
                // segments in block order are in cell order
                let mut segs = spill.segments[bucket_idx].clone();
                segs.sort_unstable_by_key(|&(seq, off, _)| (seq, off));
                let mut ordered = Vec::with_capacity(raw.len());
                for (_, off, len) in segs {
                    ordered.extend_from_slice(&raw[off as usize..(off + len) as usize]);
                }
                ordered
            } else {
                raw
            };

            let mut entries: Vec<(u32, u32, u32)> = bytes
                .par_chunks_exact(BUCKET_RECORD_LEN)
                .filter_map(|r| {
                    let word =
                        |k: usize| u32::from_le_bytes(r[k..k + 4].try_into().expect("4 bytes"));
                    Some((
                        lookup(&cell_map, word(0))?,
                        lookup(&gene_map, word(4))?,
                        word(8),
                    ))
                })
                .collect();
            drop(bytes);
            if !spill.cell_major {
                entries.par_sort_unstable();
            }

            let first_cell = bucket_idx * spill.cells_per_bucket;
            let end_cell = first_cell + spill.cells_per_bucket;
            let k0 = kept.partition_point(|&c| c < first_cell);
            let k1 = kept.partition_point(|&c| c < end_cell);
            let run_starts: Vec<usize> = (k0..=k1)
                .map(|k| entries.partition_point(|e| (e.0 as usize) < k))
                .collect();

            let built = (k0..k1)
                .into_par_iter()
                .map(|k| {
                    let j = k - k0;
                    let mut row: Vec<(u32, u32)> = entries[run_starts[j]..run_starts[j + 1]]
                        .iter()
                        .map(|&(_, g, v)| (g, v))
                        .collect();
                    compress_cell_row(&mut row, k, target_size, writer.format())
                })
                .collect::<Result<Vec<_>, ScDataErrors>>()?;

            let mut payloads = Vec::with_capacity(built.len());
            for (nnz_i, payload) in built {
                nnz.push(nnz_i);
                lib_size.push(payload.library_size);
                payloads.push(payload);
            }
            writer.write_compressed_cell_chunks(&payloads)?;

            let _ = std::fs::remove_file(temp_path);

            if verbose
                && ((bucket_idx + 1) % (n_buckets / 10).max(1) == 0 || bucket_idx + 1 == n_buckets)
            {
                println!(
                    "  Wrote bucket {}/{} ({}%)",
                    bucket_idx + 1,
                    n_buckets,
                    (bucket_idx + 1) * 100 / n_buckets
                );
            }
        }

        writer.finalise()?;

        if verbose {
            println!("Writing pass done: {:.2?}", start.elapsed());
        }

        Ok(MtxFinalData {
            cell_qc: CellQuality {
                cell_indices: kept.to_vec(),
                gene_indices: quality.genes_to_keep.to_vec(),
                lib_size,
                nnz,
            },
            no_genes: n_kept_genes,
            no_cells: n_kept_cells,
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_io::write_cell_rows;

    ///////////////
    // Fixtures //
    ///////////////

    /// Three cells x four genes, cells as rows, five entries. All indices are
    /// base-1 as the Matrix Market spec demands.
    ///
    /// ```text
    ///        g0 g1 g2 g3
    ///   c0 [  5  .  7  . ]
    ///   c1 [  .  3  .  . ]
    ///   c2 [  2  .  .  9 ]
    /// ```
    const CELLS_AS_ROWS_MTX: &str = "%%MatrixMarket matrix coordinate integer general\n\
%\n\
3 4 5\n\
1 1 5\n\
1 3 7\n\
2 2 3\n\
3 1 2\n\
3 4 9\n";

    /// The transpose of [`CELLS_AS_ROWS_MTX`], i.e. genes as rows.
    const GENES_AS_ROWS_MTX: &str = "%%MatrixMarket matrix coordinate integer general\n\
%\n\
4 3 5\n\
1 1 5\n\
3 1 7\n\
2 2 3\n\
1 3 2\n\
4 3 9\n";

    /// The `(gene indices, raw counts)` the two fixtures both encode.
    fn expected_cells() -> Vec<(Vec<u32>, Vec<u32>)> {
        vec![
            (vec![0, 2], vec![5, 7]),
            (vec![1], vec![3]),
            (vec![0, 3], vec![2, 9]),
        ]
    }

    /// RAII guard that removes a test's scratch file even if an assert fails.
    struct TempPath(PathBuf);

    /// Drop implementation for [`TempPath`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempPath {
        /// Reserve a scratch file named after the calling test. The test name
        /// is the uniqueness guarantee: `cargo test` runs the whole module in
        /// one process, so a PID alone would collide across threads.
        fn new(name: &str, ext: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_mtx_io_{name}.{ext}")))
        }

        /// Path of the guarded file as a `&str`.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Write `body` to a scratch `.mtx` file and hand back its guard.
    fn write_mtx(name: &str, body: &str) -> TempPath {
        let temp = TempPath::new(name, "mtx");
        std::fs::write(&temp.0, body).expect("mtx file written");
        temp
    }

    /// QC thresholds that keep every cell and gene of the fixtures.
    fn keep_all_qc() -> MinCellQuality {
        MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 1,
            target_size: 1e4,
        }
    }

    /// Read the first `n_cells` cells of a written store back as
    /// `(gene indices, raw counts)`.
    fn read_back(bin_path: &str, n_cells: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
        let reader = ParallelSparseReader::new(bin_path).expect("reader opens");
        let indices: Vec<usize> = (0..n_cells).collect();
        reader
            .read_cells_parallel(&indices)
            .expect("cells read back")
            .into_iter()
            .map(|c| (c.indices.clone(), c.data_raw.iter().collect()))
            .collect()
    }

    /////////////////
    // Line parser //
    /////////////////

    /// Parse an integer-field line that must not raise a value error.
    fn int(line: &[u8]) -> Option<(u32, u32, u32)> {
        parse_entry(line, MtxField::Integer).expect("no value error")
    }

    /// Regression: the parser used to saturate counts at `u16::MAX`.
    #[test]
    fn test_parse_mtx_line_keeps_full_u32_counts() {
        assert_eq!(int(b"1 2 3"), Some((1, 2, 3)));
        assert_eq!(int(b"7\t9\t65535"), Some((7, 9, 65_535)));
        assert_eq!(int(b"4 5 70000"), Some((4, 5, 70_000)));
        assert_eq!(int(b"1 2 4294967295"), Some((1, 2, u32::MAX)));
    }

    /// Empty, non-numeric and short lines give `None`, not a partial triplet.
    #[test]
    fn test_parse_mtx_line_rejects_malformed_input() {
        assert_eq!(int(b""), None);
        assert_eq!(int(b"abc"), None);
        assert_eq!(int(b"1"), None);
        assert_eq!(int(b"1 2"), None);
    }

    ////////////
    // Header //
    ////////////

    /// The shape line is split by orientation, so both readings of the same
    /// three numbers have to be pinned.
    #[test]
    fn mtx_header_shape_line_is_read_per_orientation() {
        let mtx = write_mtx("header_shape", CELLS_AS_ROWS_MTX);

        let cells_as_rows = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        assert_eq!(cells_as_rows.header.total_cells, 3);
        assert_eq!(cells_as_rows.header.total_genes, 4);
        assert_eq!(cells_as_rows.header.total_entries, 5);

        // Same file read the other way round: the first two fields swap roles.
        let genes_as_rows = MtxReader::new(mtx.path(), keep_all_qc(), false).expect("reader opens");
        assert_eq!(genes_as_rows.header.total_cells, 4);
        assert_eq!(genes_as_rows.header.total_genes, 3);
        assert_eq!(genes_as_rows.header.total_entries, 5);
    }

    /// A shape line that is not three fields must raise `MtxHeaderInvalid`
    /// rather than index past the end of the split.
    #[test]
    fn mtx_header_with_wrong_field_count_is_rejected() {
        let mtx = write_mtx(
            "header_two_fields",
            "%%MatrixMarket matrix coordinate integer general\n3 4\n1 1 5\n",
        );

        assert!(matches!(
            MtxReader::new(mtx.path(), keep_all_qc(), true),
            Err(ScDataErrors::MtxHeaderInvalid(_))
        ));
    }

    /// `MtxParseError` has to name the field that failed, and the name depends
    /// on the orientation because the first two fields swap.
    #[test]
    fn mtx_header_parse_error_names_the_failing_field() {
        let bad_second = write_mtx(
            "header_bad_second",
            "%%MatrixMarket matrix coordinate integer general\n3 x 5\n",
        );
        assert!(matches!(
            MtxReader::new(bad_second.path(), keep_all_qc(), true),
            Err(ScDataErrors::MtxParseError {
                field: "gene count"
            })
        ));
        // Cells are the columns now, so the same broken field is the cell count.
        assert!(matches!(
            MtxReader::new(bad_second.path(), keep_all_qc(), false),
            Err(ScDataErrors::MtxParseError {
                field: "cell count"
            })
        ));

        let bad_third = write_mtx(
            "header_bad_third",
            "%%MatrixMarket matrix coordinate integer general\n3 4 x\n",
        );
        assert!(matches!(
            MtxReader::new(bad_third.path(), keep_all_qc(), true),
            Err(ScDataErrors::MtxParseError {
                field: "entry count"
            })
        ));
    }

    /// A missing file must surface as an I/O error, not a panic on unwrap.
    #[test]
    fn mtx_reader_reports_a_missing_file() {
        let missing = std::env::temp_dir().join("bixverse_mtx_io_does_not_exist.mtx");
        assert!(matches!(
            MtxReader::new(&missing, keep_all_qc(), true),
            Err(ScDataErrors::BinaryIo(_))
        ));
    }

    /////////////////
    // Quality pass //
    /////////////////

    /// `parse_mtx_quality` drives every downstream index remap, so pin the
    /// gene-then-cell filter order and the old-to-new maps it builds.
    #[test]
    fn parse_mtx_quality_filters_genes_then_cells() {
        let mtx = write_mtx("quality_filters", CELLS_AS_ROWS_MTX);

        // Everything permissive: nothing is dropped and the maps are identities.
        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        assert_eq!(quality.cells_to_keep, vec![0, 1, 2]);
        assert_eq!(quality.genes_to_keep, vec![0, 1, 2, 3]);

        // min_cells = 2 keeps only gene 0, which is seen in cells 0 and 2.
        // Cell 1 then has zero kept entries and fails min_unique_genes = 1.
        let qc = MinCellQuality {
            min_unique_genes: 1,
            min_lib_size: 1,
            min_cells: 2,
            target_size: 1e4,
        };
        let mut reader = MtxReader::new(mtx.path(), qc, true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        assert_eq!(quality.genes_to_keep, vec![0]);
        assert_eq!(quality.cells_to_keep, vec![0, 2]);
        assert_eq!(quality.cell_old_to_new[&0], 0);
        assert_eq!(quality.cell_old_to_new[&2], 1);
        assert_eq!(quality.gene_old_to_new[&0], 0);
    }

    /////////////////
    // Round trips //
    /////////////////

    /// The only mtx -> bin -> reader path in the crate. Counts and gene indices
    /// have to survive the conversion untouched.
    #[test]
    fn mtx_round_trip_preserves_counts_and_indices() {
        let mtx = write_mtx("round_trip", CELLS_AS_ROWS_MTX);
        let bin = TempPath::new("round_trip", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let final_data = reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        assert_eq!(final_data.no_cells, 3);
        assert_eq!(final_data.no_genes, 4);
        // Library sizes: 5 + 7 = 12, 3, 2 + 9 = 11.
        assert_eq!(final_data.cell_qc.lib_size, vec![12, 3, 11]);
        assert_eq!(final_data.cell_qc.nnz, vec![2, 1, 2]);

        assert_eq!(read_back(bin.path(), 3), expected_cells());
    }

    /// Matrix Market is base-1. The lowest legal index has to land on 0 and the
    /// highest on `n - 1`, with nothing shifted in between.
    #[test]
    fn mtx_indices_shift_from_base_one_to_base_zero() {
        let mtx = write_mtx(
            "base_one",
            "%%MatrixMarket matrix coordinate integer general\n2 3 2\n1 1 42\n2 3 7\n",
        );
        let bin = TempPath::new("base_one", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        // Only genes 1 and 3 (base-1) carry counts, so the kept genes are the
        // old 0 and 2, renumbered to 0 and 1.
        assert_eq!(quality.genes_to_keep, vec![0, 2]);
        assert_eq!(
            read_back(bin.path(), 2),
            vec![(vec![0], vec![42]), (vec![1], vec![7])]
        );
    }

    /// The two orientations parse different files but must produce the same
    /// store, which is the only cross-check on the row/column swap.
    #[test]
    fn mtx_gene_as_rows_matches_cell_as_rows() {
        let mtx = write_mtx("orientation", GENES_AS_ROWS_MTX);
        let bin = TempPath::new("orientation", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), false).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let final_data = reader
            .process_mtx_and_write_bin(bin.path(), &quality, false)
            .expect("conversion");

        assert_eq!(final_data.no_cells, 3);
        assert_eq!(final_data.no_genes, 4);
        assert_eq!(final_data.cell_qc.lib_size, vec![12, 3, 11]);
        assert_eq!(read_back(bin.path(), 3), expected_cells());
    }

    /// The streaming writer buckets to temp files and re-sorts; it must land on
    /// exactly the same store as the in-memory writer.
    #[test]
    fn mtx_streaming_matches_non_streaming() {
        let mtx = write_mtx("streaming", CELLS_AS_ROWS_MTX);
        let bin_plain = TempPath::new("streaming_plain", "bin");
        let bin_stream = TempPath::new("streaming_stream", "bin");

        let mut reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let plain = reader
            .process_mtx_and_write_bin(bin_plain.path(), &quality, false)
            .expect("conversion");

        let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let streamed = reader
            .process_mtx_and_write_bin_streaming(bin_stream.path(), &quality, false)
            .expect("streaming conversion");

        assert_eq!(streamed.no_cells, plain.no_cells);
        assert_eq!(streamed.no_genes, plain.no_genes);
        assert_eq!(streamed.cell_qc.lib_size, plain.cell_qc.lib_size);
        assert_eq!(streamed.cell_qc.nnz, plain.cell_qc.nnz);
        assert_eq!(
            read_back(bin_stream.path(), 3),
            read_back(bin_plain.path(), 3)
        );
    }

    /////////////////////
    // Reference store //
    /////////////////////

    /// Entry order of a generated mtx file.
    #[derive(Clone, Copy, Debug)]
    enum EntryOrder {
        /// All entries of a cell together, cells ascending
        CellMajor,
        /// All entries of a gene together, genes ascending
        GeneMajor,
        /// Seeded shuffle
        Shuffled,
    }

    /// Random sparse counts with rare genes and weak cells, so QC drops both.
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells
    /// * `n_genes` - Number of genes
    /// * `seed` - Seed
    ///
    /// ### Returns
    ///
    /// Unique `(cell, gene, count)` triplets, 0-based, cell-major.
    fn random_counts(n_cells: usize, n_genes: usize, seed: u64) -> Vec<(u32, u32, u32)> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::new();
        for c in 0..n_cells {
            let k = rng.random_range(0..40usize);
            let mut genes: Vec<u32> = (0..k)
                .map(|_| {
                    // squared uniform skews towards low indices, leaving the
                    // top of the range rare
                    let u: f64 = rng.random();
                    ((u * u * u) * n_genes as f64) as u32
                })
                .collect();
            genes.sort_unstable();
            genes.dedup();
            for g in genes {
                let v = if rng.random_range(0..200) == 0 {
                    70_000
                } else {
                    rng.random_range(1..30)
                };
                out.push((c as u32, g, v));
            }
        }
        out
    }

    /// Render triplets as an mtx file body.
    ///
    /// ### Params
    ///
    /// * `entries` - `(cell, gene, count)`, 0-based
    /// * `n_cells` - Number of cells
    /// * `n_genes` - Number of genes
    /// * `cells_as_rows` - Orientation
    /// * `order` - Entry order
    ///
    /// ### Returns
    ///
    /// The file text.
    fn render_mtx(
        entries: &[(u32, u32, u32)],
        n_cells: usize,
        n_genes: usize,
        cells_as_rows: bool,
        order: EntryOrder,
    ) -> String {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        let mut e = entries.to_vec();
        match order {
            EntryOrder::CellMajor => {}
            EntryOrder::GeneMajor => e.sort_by_key(|&(c, g, _)| (g, c)),
            EntryOrder::Shuffled => e.shuffle(&mut StdRng::seed_from_u64(7)),
        }
        let (rows, cols) = if cells_as_rows {
            (n_cells, n_genes)
        } else {
            (n_genes, n_cells)
        };
        let mut s = format!(
            "%%MatrixMarket matrix coordinate integer general\n%\n{rows} {cols} {}\n",
            e.len()
        );
        for (c, g, v) in e {
            let (r, k) = if cells_as_rows { (c, g) } else { (g, c) };
            s.push_str(&format!("{} {} {v}\n", r + 1, k + 1));
        }
        s
    }

    /// Independent reference loader: QC and store written straight from the
    /// triplets through [`write_cell_rows`].
    ///
    /// ### Params
    ///
    /// * `entries` - `(cell, gene, count)`, 0-based
    /// * `n_cells` - Number of cells
    /// * `n_genes` - Number of genes
    /// * `qc` - Thresholds
    /// * `bin_path` - Store to write
    ///
    /// ### Returns
    ///
    /// Kept cells and kept genes.
    fn reference_store(
        entries: &[(u32, u32, u32)],
        n_cells: usize,
        n_genes: usize,
        qc: &MinCellQuality,
        bin_path: &str,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut gene_nnz = vec![0usize; n_genes];
        for &(_, g, _) in entries {
            gene_nnz[g as usize] += 1;
        }
        let genes: Vec<usize> = (0..n_genes)
            .filter(|&g| gene_nnz[g] >= qc.min_cells)
            .collect();
        let gene_map = dense_index_map(&genes);

        let mut stats = vec![(0usize, 0u64); n_cells];
        for &(c, g, v) in entries {
            if gene_map
                .get(g as usize)
                .is_some_and(|&m| m != INDEX_DROPPED)
            {
                stats[c as usize].0 += 1;
                stats[c as usize].1 += v as u64;
            }
        }
        let cells: Vec<usize> = (0..n_cells)
            .filter(|&c| {
                stats[c].0 >= qc.min_unique_genes && stats[c].1 as f32 >= qc.min_lib_size as f32
            })
            .collect();
        let cell_map = dense_index_map(&cells);

        let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); cells.len()];
        for &(c, g, v) in entries {
            let (Some(&nc), Some(&ng)) = (cell_map.get(c as usize), gene_map.get(g as usize))
            else {
                continue;
            };
            if nc != INDEX_DROPPED && ng != INDEX_DROPPED {
                rows[nc as usize].push((ng, v));
            }
        }
        let mut writer =
            CellGeneSparseWriter::new(bin_path, true, cells.len(), genes.len(), qc.target_size)
                .expect("writer");
        write_cell_rows(&mut rows, 0, qc.target_size, &mut writer).expect("rows written");
        writer.finalise().expect("finalised");
        (cells, genes)
    }

    /// QC that drops some genes and some cells of [`random_counts`].
    fn dropping_qc() -> MinCellQuality {
        MinCellQuality {
            min_unique_genes: 8,
            min_lib_size: 60,
            min_cells: 3,
            target_size: 1e4,
        }
    }

    /// The loader's store must be byte identical to the reference for both
    /// orientations and every entry order, with QC dropping cells and genes.
    #[test]
    fn mtx_store_matches_reference_for_every_layout() {
        let (n_cells, n_genes) = (400, 1_500);
        let entries = random_counts(n_cells, n_genes, 42);
        let qc = dropping_qc();
        let reference = TempPath::new("reference_store", "bin");
        let (cells, genes) = reference_store(&entries, n_cells, n_genes, &qc, reference.path());
        assert!(
            cells.len() < n_cells && genes.len() < n_genes,
            "QC drops both"
        );
        let expected = std::fs::read(&reference.0).expect("reference store");

        for cells_as_rows in [true, false] {
            for order in [
                EntryOrder::CellMajor,
                EntryOrder::GeneMajor,
                EntryOrder::Shuffled,
            ] {
                let body = render_mtx(&entries, n_cells, n_genes, cells_as_rows, order);
                // (gzip, block size): one block, and blocks small enough that
                // cells straddle them and the pipeline sees many
                for (gz, block_bytes) in [(false, MTX_BLOCK_BYTES), (false, 101), (true, 101)] {
                    let name = format!("layout_{cells_as_rows}_{order:?}_{gz}_{block_bytes}");
                    let mtx = if gz {
                        write_gz(&name, &body, 2)
                    } else {
                        write_mtx(&name, &body)
                    };
                    let bin = TempPath::new(&name, "bin");

                    let mut reader = MtxReader::new(mtx.path(), qc.clone(), cells_as_rows)
                        .expect("reader opens");
                    reader.source.block_bytes = block_bytes;
                    let quality = reader.parse_mtx_quality(false).expect("quality pass");
                    assert_eq!(quality.cells_to_keep, cells, "{name}: kept cells");
                    assert_eq!(quality.genes_to_keep, genes, "{name}: kept genes");
                    let spill = reader.spill.as_ref().expect("spilled");
                    assert_eq!(
                        spill.cell_major,
                        matches!(order, EntryOrder::CellMajor),
                        "{name}: order detection"
                    );
                    reader
                        .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
                        .expect("conversion");
                    let got = std::fs::read(&bin.0).expect("store");
                    assert!(got == expected, "{name}: store differs from the reference");
                }
            }
        }
    }

    ////////////////
    // Field type //
    ////////////////

    /// Write `body` as a gzip of `members` concatenated members, as bgzip or
    /// `cat a.gz b.gz` produce.
    fn write_gz(name: &str, body: &str, members: usize) -> TempPath {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        let temp = TempPath::new(name, "mtx.gz");
        let bytes = body.as_bytes();
        let step = bytes.len().div_ceil(members.max(1)).max(1);
        let mut out = Vec::new();
        for part in bytes.chunks(step) {
            let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
            enc.write_all(part).expect("gzip member");
            out.extend(enc.finish().expect("gzip member"));
        }
        std::fs::write(&temp.0, out).expect("gz file written");
        temp
    }

    /// Integral reals, scientific notation included, parse as counts; a
    /// fraction, a negative or garbage is an error, never a truncation.
    #[test]
    fn real_values_must_be_integral_counts() {
        let real = |l: &[u8]| parse_entry(l, MtxField::Real);
        assert_eq!(real(b"1 2 3").unwrap(), Some((1, 2, 3)));
        assert_eq!(real(b"1 2 3.0").unwrap(), Some((1, 2, 3)));
        assert_eq!(real(b"1 2 1.5e+01").unwrap(), Some((1, 2, 15)));
        assert_eq!(real(b"1 2 2.000000e+00").unwrap(), Some((1, 2, 2)));
        assert_eq!(real(b"1 2 0").unwrap(), Some((1, 2, 0)));
        for bad in [
            &b"1 2 1.5"[..],
            b"1 2 -1",
            b"1 2 abc",
            b"1 2 1e300",
            b"1 2 nan",
        ] {
            assert!(
                matches!(real(bad), Err(ScDataErrors::MtxNotACount { .. })),
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
        // an integer field with a fractional value is just as wrong
        assert!(matches!(
            parse_entry(b"1 2 1.5", MtxField::Integer),
            Err(ScDataErrors::MtxNotACount { .. })
        ));
    }

    /// A pattern entry has no value column and counts 1.
    #[test]
    fn pattern_entries_count_one() {
        assert_eq!(
            parse_entry(b"3 4", MtxField::Pattern).unwrap(),
            Some((3, 4, 1))
        );
        assert_eq!(
            parse_entry(b"3 4 9", MtxField::Pattern).unwrap(),
            Some((3, 4, 1))
        );
        assert_eq!(parse_entry(b"3", MtxField::Pattern).unwrap(), None);
    }

    /// The banner decides the field; complex, dense and symmetric files are
    /// rejected up front.
    #[test]
    fn mtx_banner_field_and_symmetry() {
        let shape = "3 4 5\n1 1 5\n";
        let open = |name: &str, banner: &str| {
            let mtx = write_mtx(name, &format!("{banner}\n{shape}"));
            MtxReader::new(mtx.path(), keep_all_qc(), true).map(|r| r.header.field)
        };
        assert_eq!(
            open(
                "banner_int",
                "%%MatrixMarket matrix coordinate integer general"
            )
            .unwrap(),
            MtxField::Integer
        );
        assert_eq!(
            open(
                "banner_real",
                "%%MatrixMarket Matrix Coordinate Real General"
            )
            .unwrap(),
            MtxField::Real
        );
        assert_eq!(
            open(
                "banner_pattern",
                "%%MatrixMarket matrix coordinate pattern general"
            )
            .unwrap(),
            MtxField::Pattern
        );
        assert_eq!(
            open("banner_none", "% no banner").unwrap(),
            MtxField::Integer
        );
        for (name, banner) in [
            (
                "banner_complex",
                "%%MatrixMarket matrix coordinate complex general",
            ),
            (
                "banner_sym",
                "%%MatrixMarket matrix coordinate integer symmetric",
            ),
            (
                "banner_array",
                "%%MatrixMarket matrix array integer general",
            ),
        ] {
            assert!(
                matches!(open(name, banner), Err(ScDataErrors::MtxUnsupported(_))),
                "{name}"
            );
        }
    }

    /// A real file of integral values and a pattern file load end to end, and
    /// a real file with a fraction fails the quality pass.
    #[test]
    fn mtx_real_and_pattern_files_round_trip() {
        let real = write_mtx(
            "real_file",
            "%%MatrixMarket matrix coordinate real general\n2 3 3\n1 1 5.0\n1 3 1.2e+01\n2 2 3\n",
        );
        let bin = TempPath::new("real_file", "bin");
        let mut reader = MtxReader::new(real.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        let data = reader
            .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            .expect("conversion");
        assert_eq!(data.cell_qc.lib_size, vec![17, 3]);
        assert_eq!(
            read_back(bin.path(), 2),
            vec![(vec![0, 2], vec![5, 12]), (vec![1], vec![3])]
        );

        let pattern = write_mtx(
            "pattern_file",
            "%%MatrixMarket matrix coordinate pattern general\n2 3 3\n1 1\n1 3\n2 2\n",
        );
        let bin = TempPath::new("pattern_file", "bin");
        let mut reader = MtxReader::new(pattern.path(), keep_all_qc(), true).expect("reader opens");
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        reader
            .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            .expect("conversion");
        assert_eq!(
            read_back(bin.path(), 2),
            vec![(vec![0, 2], vec![1, 1]), (vec![1], vec![1])]
        );

        let fraction = write_mtx(
            "real_fraction",
            "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 5.0\n2 2 0.25\n",
        );
        let mut reader =
            MtxReader::new(fraction.path(), keep_all_qc(), true).expect("reader opens");
        assert!(matches!(
            reader.parse_mtx_quality(false),
            Err(ScDataErrors::MtxNotACount { .. })
        ));
    }

    /// A gzipped file is recognised by its magic bytes whatever its name.
    #[test]
    fn mtx_gzip_is_detected_by_magic_bytes() {
        let gz = write_gz("gz_magic", CELLS_AS_ROWS_MTX, 1);
        let renamed = TempPath::new("gz_magic_renamed", "mtx");
        std::fs::copy(&gz.0, &renamed.0).expect("copy");
        let bin = TempPath::new("gz_magic", "bin");
        let mut reader = MtxReader::new(renamed.path(), keep_all_qc(), true).expect("reader opens");
        assert!(reader.source.gz);
        let quality = reader.parse_mtx_quality(false).expect("quality pass");
        reader
            .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            .expect("conversion");
        assert_eq!(read_back(bin.path(), 3), expected_cells());
    }

    /// An empty input has no cells to keep; the streaming path short-circuits
    /// there and still has to leave a readable, empty store behind.
    #[test]
    fn mtx_streaming_writes_a_readable_store_when_nothing_is_kept() {
        let bin = TempPath::new("streaming_empty", "bin");
        let mtx = write_mtx("streaming_empty", CELLS_AS_ROWS_MTX);

        let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
        let mut quality = CellOnFileQuality::new(vec![], vec![0, 1, 2, 3]);
        quality.generate_maps_sets();

        let final_data = reader
            .process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            .expect("streaming conversion");

        assert_eq!(final_data.no_cells, 0);
        assert_eq!(final_data.no_genes, 4);
        assert!(final_data.cell_qc.lib_size.is_empty());

        let reader = ParallelSparseReader::new(bin.path()).expect("reader opens");
        assert!(reader.is_cell_based());
        assert_eq!(reader.get_header().total_cells, 0);
    }

    /// A kept cell with no kept entries is still written, as an empty chunk,
    /// so `lib_size` / `nnz` line up with `cell_indices` in both writers.
    #[test]
    fn mtx_kept_cell_without_entries_is_written_empty() {
        for streaming in [false, true] {
            let name = format!("empty_cell_{streaming}");
            let mtx = write_mtx(&name, CELLS_AS_ROWS_MTX);
            let bin = TempPath::new(&name, "bin");

            // Keep every cell but only gene 0, which cell 1 does not express.
            let reader = MtxReader::new(mtx.path(), keep_all_qc(), true).expect("reader opens");
            let mut quality = CellOnFileQuality::new(vec![0, 1, 2], vec![0]);
            quality.generate_maps_sets();

            let final_data = if streaming {
                reader.process_mtx_and_write_bin_streaming(bin.path(), &quality, false)
            } else {
                reader.process_mtx_and_write_bin(bin.path(), &quality, false)
            }
            .expect("conversion");

            assert_eq!(final_data.cell_qc.cell_indices.len(), 3);
            assert_eq!(final_data.cell_qc.lib_size, vec![5, 0, 2]);
            assert_eq!(final_data.cell_qc.nnz, vec![1, 0, 1]);

            let store = ParallelSparseReader::new(bin.path()).expect("reader opens");
            let cell = store
                .read_cells_parallel(&[1])
                .expect("empty cell is on disk");
            assert!(cell[0].indices.is_empty());
        }
    }
}
