//! Parallel reads of gzip-compressed 1-D HDF5 datasets.
//!
//! `H5Dread` inflates every chunk on one core under the HDF5 global lock, so
//! the h5ad and 10x readers sit at one busy core no matter how many threads
//! ask. For the common case (chunked, rank 1, little-endian numbers, filter
//! pipeline `[deflate]` or `[shuffle, deflate]`) the chunk index is read once
//! through HDF5 and the chunks themselves are read with plain positional
//! reads, inflated and unshuffled in parallel off the lock. Anything else
//! falls back to `H5Dread`.
//!
//! The chunk index comes from `H5Dchunk_iter` when the linked HDF5 has it
//! (1.14+), else from `H5Dget_chunk_info_by_coord` (1.10.5+). Older libraries
//! always take the fallback. `H5Dget_chunk_info` by index is quadratic in the
//! number of chunks and is never used.

use hdf5::Dataset;
use hdf5::filters::Filter;
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use rayon::prelude::*;
use std::ops::Range;

use crate::errors::ScDataErrors;

///////////////////
// Element types //
///////////////////

/// On-disk element type of an eligible dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stored {
    /// Signed 8 bit integer
    I8,
    /// Signed 16 bit integer
    I16,
    /// Signed 32 bit integer
    I32,
    /// Signed 64 bit integer
    I64,
    /// Unsigned 8 bit integer
    U8,
    /// Unsigned 16 bit integer
    U16,
    /// Unsigned 32 bit integer
    U32,
    /// Unsigned 64 bit integer
    U64,
    /// 32 bit float
    F32,
    /// 64 bit float
    F64,
}

impl Stored {
    /// Map an HDF5 type descriptor to a stored type.
    ///
    /// ### Params
    ///
    /// * `desc` - The dataset's type descriptor
    ///
    /// ### Returns
    ///
    /// The stored type, or `None` for anything that is not a plain number.
    fn from_descriptor(desc: &TypeDescriptor) -> Option<Self> {
        Some(match desc {
            TypeDescriptor::Integer(IntSize::U1) => Self::I8,
            TypeDescriptor::Integer(IntSize::U2) => Self::I16,
            TypeDescriptor::Integer(IntSize::U4) => Self::I32,
            TypeDescriptor::Integer(IntSize::U8) => Self::I64,
            TypeDescriptor::Unsigned(IntSize::U1) => Self::U8,
            TypeDescriptor::Unsigned(IntSize::U2) => Self::U16,
            TypeDescriptor::Unsigned(IntSize::U4) => Self::U32,
            TypeDescriptor::Unsigned(IntSize::U8) => Self::U64,
            TypeDescriptor::Float(FloatSize::U4) => Self::F32,
            TypeDescriptor::Float(FloatSize::U8) => Self::F64,
            _ => return None,
        })
    }

    /// Element size in bytes.
    ///
    /// ### Returns
    ///
    /// The size of one element on disk.
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// Element types the reader can hand back.
///
/// The conversions follow HDF5's hard conversions for values in range:
/// integers clamp into the target range, floats truncate towards zero and
/// saturate when converted to integers.
pub trait H5Elem: hdf5::H5Type + Copy + Default + Send + Sync {
    /// Convert from a signed integer.
    fn from_i64(v: i64) -> Self;
    /// Convert from an unsigned integer.
    fn from_u64(v: u64) -> Self;
    /// Convert from a float.
    fn from_f64(v: f64) -> Self;
}

impl H5Elem for f32 {
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as f32
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v as f32
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl H5Elem for u32 {
    #[inline]
    fn from_i64(v: i64) -> Self {
        v.clamp(0, u32::MAX as i64) as u32
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v.min(u32::MAX as u64) as u32
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u32
    }
}

impl H5Elem for i64 {
    #[inline]
    fn from_i64(v: i64) -> Self {
        v
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v.min(i64::MAX as u64) as i64
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i64
    }
}

impl H5Elem for usize {
    #[inline]
    fn from_i64(v: i64) -> Self {
        v.max(0) as usize
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v as usize
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as usize
    }
}

/// Decode little-endian bytes of the stored type into `out`.
///
/// ### Params
///
/// * `bytes` - Exactly `out.len() * stored.size()` bytes
/// * `stored` - The on-disk element type
/// * `out` - Destination
fn convert_into<T: H5Elem>(bytes: &[u8], stored: Stored, out: &mut [T]) {
    macro_rules! conv {
        ($t:ty, $n:expr, $via:ident, $wide:ty) => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact($n)) {
                let v = <$t>::from_le_bytes(b.try_into().expect("chunks_exact width"));
                *o = T::$via(v as $wide);
            }
        };
    }
    match stored {
        Stored::I8 => conv!(i8, 1, from_i64, i64),
        Stored::I16 => conv!(i16, 2, from_i64, i64),
        Stored::I32 => conv!(i32, 4, from_i64, i64),
        Stored::I64 => conv!(i64, 8, from_i64, i64),
        Stored::U8 => conv!(u8, 1, from_u64, u64),
        Stored::U16 => conv!(u16, 2, from_u64, u64),
        Stored::U32 => conv!(u32, 4, from_u64, u64),
        Stored::U64 => conv!(u64, 8, from_u64, u64),
        Stored::F32 => conv!(f32, 4, from_f64, f64),
        Stored::F64 => conv!(f64, 8, from_f64, f64),
    }
}

/////////////////
// Chunk index //
/////////////////

/// Location of one chunk in the file.
#[derive(Clone, Copy, Debug)]
struct ChunkLoc {
    /// Absolute byte address of the stored chunk
    addr: u64,
    /// Stored (compressed) size in bytes
    size: u64,
    /// Bit `i` set means filter `i` of the pipeline was skipped for this chunk
    filter_mask: u32,
}

/// Everything needed to read chunks without going through `H5Dread`.
struct DirectChunks {
    /// The same file, opened again for positional reads
    file: std::fs::File,
    /// Number of elements in the dataset
    len: usize,
    /// Elements per chunk
    chunk_len: usize,
    /// On-disk element type
    stored: Stored,
    /// Whether the pipeline starts with shuffle (then deflate is filter 1)
    shuffle: bool,
    /// One entry per chunk, in logical order
    chunks: Vec<ChunkLoc>,
}

/// Collect the chunk locations of a 1-D dataset.
///
/// ### Params
///
/// * `ds` - The dataset
/// * `n_chunks` - Expected number of chunks
/// * `chunk_len` - Elements per chunk
///
/// ### Returns
///
/// One location per chunk in logical order, or `None` if a chunk is missing
/// (never written) or the linked HDF5 cannot report chunk locations.
#[cfg(hdf5_chunk_iter)]
fn chunk_index(ds: &Dataset, n_chunks: usize, chunk_len: usize) -> Option<Vec<ChunkLoc>> {
    let mut found: Vec<(u64, ChunkLoc)> = Vec::with_capacity(n_chunks);
    ds.chunks_visit(|c| {
        found.push((
            c.offset[0],
            ChunkLoc {
                addr: c.addr,
                size: c.size,
                filter_mask: c.filter_mask,
            },
        ));
        0
    })
    .ok()?;
    found.sort_unstable_by_key(|&(off, _)| off);
    let complete = found.len() == n_chunks
        && found
            .iter()
            .enumerate()
            .all(|(i, &(off, _))| off == (i * chunk_len) as u64);
    complete.then(|| found.into_iter().map(|(_, loc)| loc).collect())
}

/// Collect the chunk locations of a 1-D dataset.
///
/// ### Params
///
/// * `ds` - The dataset
/// * `n_chunks` - Expected number of chunks
/// * `chunk_len` - Elements per chunk
///
/// ### Returns
///
/// One location per chunk in logical order, or `None` if a chunk is missing
/// (never written) or the linked HDF5 cannot report chunk locations.
#[cfg(all(hdf5_chunk_info, not(hdf5_chunk_iter)))]
fn chunk_index(ds: &Dataset, n_chunks: usize, chunk_len: usize) -> Option<Vec<ChunkLoc>> {
    use hdf5_sys::h5::HADDR_UNDEF;
    use hdf5_sys::h5d::H5Dget_chunk_info_by_coord;

    let id = ds.id();
    hdf5::sync::sync(|| {
        (0..n_chunks)
            .map(|i| {
                let offset = (i * chunk_len) as hdf5_sys::h5::hsize_t;
                let mut filter_mask = 0;
                let mut addr = 0;
                let mut size = 0;
                // SAFETY: `id` is a live dataset of rank 1 and every out
                // pointer refers to a local; the global HDF5 lock is held.
                let status = unsafe {
                    H5Dget_chunk_info_by_coord(id, &offset, &mut filter_mask, &mut addr, &mut size)
                };
                (status >= 0 && addr != HADDR_UNDEF).then_some(ChunkLoc {
                    addr,
                    size,
                    filter_mask,
                })
            })
            .collect()
    })
}

/// Collect the chunk locations of a 1-D dataset.
///
/// The linked HDF5 predates 1.10.5 and cannot report chunk locations.
///
/// ### Returns
///
/// Always `None`.
#[cfg(not(any(hdf5_chunk_info, hdf5_chunk_iter)))]
fn chunk_index(_ds: &Dataset, _n_chunks: usize, _chunk_len: usize) -> Option<Vec<ChunkLoc>> {
    None
}

/// Decide whether a dataset can be read directly and build its chunk index.
///
/// ### Params
///
/// * `ds` - The dataset
///
/// ### Returns
///
/// The direct reader state, or `None` if any requirement fails.
fn probe(ds: &Dataset) -> Option<DirectChunks> {
    if !ds.is_chunked() || ds.ndim() != 1 {
        return None;
    }
    let dtype = ds.dtype().ok()?;
    if dtype.byte_order() != hdf5::datatype::ByteOrder::LittleEndian {
        return None;
    }
    let stored = Stored::from_descriptor(&dtype.to_descriptor().ok()?)?;
    if dtype.size() != stored.size() {
        return None;
    }
    let shuffle = match ds.dcpl().ok()?.get_filters().ok()?.as_slice() {
        [Filter::Deflate(_)] => false,
        [Filter::Shuffle, Filter::Deflate(_)] => true,
        _ => return None,
    };

    // chunk addresses are relative to the HDF5 base address, which a user
    // block moves away from the start of the file
    let file = ds.file().ok()?;
    if file.userblock() != 0
        || !matches!(
            file.fapl().ok()?.driver(),
            hdf5::file::FileDriver::Sec2 | hdf5::file::FileDriver::Stdio
        )
    {
        return None;
    }

    let chunk_len = *ds.chunk()?.first()?;
    let len = ds.size();
    if chunk_len == 0 {
        return None;
    }
    let chunks = chunk_index(ds, len.div_ceil(chunk_len), chunk_len)?;
    let file = std::fs::File::open(file.filename()).ok()?;

    Some(DirectChunks {
        file,
        len,
        chunk_len,
        stored,
        shuffle,
        chunks,
    })
}

//////////////
// Decoding //
//////////////

/// Per-thread buffers for one chunk.
#[derive(Default)]
struct Scratch {
    /// Bytes as stored in the file
    stored: Vec<u8>,
    /// Decoded bytes
    decoded: Vec<u8>,
}

/// Read `buf.len()` bytes at `offset`.
///
/// ### Params
///
/// * `file` - The file
/// * `buf` - Destination, filled completely
/// * `offset` - Absolute byte offset
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Read `buf.len()` bytes at `offset`.
///
/// `seek_read` moves the shared cursor on Windows, which is harmless here
/// since every read names its own offset.
///
/// ### Params
///
/// * `file` - The file
/// * `buf` - Destination, filled completely
/// * `offset` - Absolute byte offset
#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Inflate a zlib stream of known decoded size.
///
/// ### Params
///
/// * `src` - The zlib stream, as the HDF5 deflate filter writes it
/// * `dst` - Destination, exactly the decoded size
///
/// ### Returns
///
/// `true` if the stream inflated to exactly `dst.len()` bytes.
fn inflate(src: &[u8], dst: &mut [u8]) -> bool {
    let mut stream = flate2::Decompress::new(true);
    matches!(
        stream.decompress(src, dst, flate2::FlushDecompress::Finish),
        Ok(flate2::Status::StreamEnd)
    ) && stream.total_out() as usize == dst.len()
}

/// Undo the HDF5 shuffle filter.
///
/// Shuffle stores byte `j` of every element contiguously, for each `j`, and
/// copies any trailing bytes that do not fill an element verbatim.
///
/// ### Params
///
/// * `src` - Shuffled bytes
/// * `dst` - Destination, same length as `src`
/// * `size` - Element size in bytes
fn unshuffle(src: &[u8], dst: &mut [u8], size: usize) {
    let n = src.len() / size;
    for (i, el) in dst[..n * size].chunks_exact_mut(size).enumerate() {
        for (j, b) in el.iter_mut().enumerate() {
            *b = src[j * n + i];
        }
    }
    dst[n * size..].copy_from_slice(&src[n * size..]);
}

impl DirectChunks {
    /// Read and decode one chunk.
    ///
    /// ### Params
    ///
    /// * `idx` - Chunk number
    /// * `s` - Thread-local scratch
    ///
    /// ### Returns
    ///
    /// The decoded chunk bytes, `chunk_len * stored.size()` long.
    fn decode<'a>(&self, idx: usize, s: &'a mut Scratch) -> Result<&'a [u8], ScDataErrors> {
        let loc = self.chunks[idx];
        let n_bytes = self.chunk_len * self.stored.size();
        let deflate_bit = if self.shuffle { 1 } else { 0 };
        let deflated = loc.filter_mask & (1 << deflate_bit) == 0;
        let shuffled = self.shuffle && loc.filter_mask & 1 == 0;

        s.stored.resize(loc.size as usize, 0);
        read_exact_at(&self.file, &mut s.stored, loc.addr)?;

        if deflated {
            s.decoded.resize(n_bytes, 0);
            if !inflate(&s.stored, &mut s.decoded) {
                return Err(ScDataErrors::H5ChunkDecode(format!(
                    "chunk {idx} did not inflate to {n_bytes} bytes"
                )));
            }
            std::mem::swap(&mut s.stored, &mut s.decoded);
        }
        if s.stored.len() != n_bytes {
            return Err(ScDataErrors::H5ChunkDecode(format!(
                "chunk {idx} holds {} bytes, expected {n_bytes}",
                s.stored.len()
            )));
        }
        if shuffled {
            s.decoded.resize(n_bytes, 0);
            unshuffle(&s.stored, &mut s.decoded, self.stored.size());
            std::mem::swap(&mut s.stored, &mut s.decoded);
        }
        Ok(&s.stored)
    }

    /// Read a range of elements, decoding the overlapping chunks in parallel.
    ///
    /// ### Params
    ///
    /// * `range` - Element range, within the dataset
    ///
    /// ### Returns
    ///
    /// The elements converted to `T`.
    fn read<T: H5Elem>(&self, range: Range<usize>) -> Result<Vec<T>, ScDataErrors> {
        let Range { start, end } = range;
        if start > end || end > self.len {
            return Err(ScDataErrors::H5ChunkDecode(format!(
                "range {start}..{end} out of bounds for {} elements",
                self.len
            )));
        }
        let mut out = vec![T::default(); end - start];
        if start == end {
            return Ok(out);
        }

        let cl = self.chunk_len;
        let mut pieces = Vec::with_capacity((end - 1) / cl - start / cl + 1);
        let mut rest = out.as_mut_slice();
        let mut pos = start;
        while pos < end {
            let idx = pos / cl;
            let piece_end = ((idx + 1) * cl).min(end);
            let (head, tail) = rest.split_at_mut(piece_end - pos);
            pieces.push((idx, pos - idx * cl, head));
            rest = tail;
            pos = piece_end;
        }

        let size = self.stored.size();
        pieces
            .into_par_iter()
            .try_for_each_init(Scratch::default, |s, (idx, skip, dst)| {
                let bytes = self.decode(idx, s)?;
                convert_into(
                    &bytes[skip * size..(skip + dst.len()) * size],
                    self.stored,
                    dst,
                );
                Ok::<(), ScDataErrors>(())
            })?;
        Ok(out)
    }
}

////////////
// Reader //
////////////

/// 1-D dataset reader that decodes gzip chunks in parallel where it can.
///
/// Construct once per dataset and pass, by reference, to every thread that
/// reads from it: building it reads the chunk index. Safe to share; the
/// direct path never touches HDF5 after construction.
pub struct H5ChunkedReader {
    /// The dataset, kept for the `H5Dread` fallback
    ds: Dataset,
    /// Direct reader state, `None` when the dataset is not eligible
    direct: Option<DirectChunks>,
}

impl H5ChunkedReader {
    /// Open a reader on a dataset.
    ///
    /// ### Params
    ///
    /// * `ds` - A 1-D dataset
    ///
    /// ### Returns
    ///
    /// The reader. Ineligible datasets read through `H5Dread`.
    pub fn new(ds: Dataset) -> Self {
        let direct = probe(&ds);
        Self { ds, direct }
    }

    /// Whether reads bypass `H5Dread`.
    ///
    /// ### Returns
    ///
    /// `true` if the dataset passed every eligibility check.
    pub fn is_direct(&self) -> bool {
        self.direct.is_some()
    }

    /// Read a range of elements.
    ///
    /// ### Params
    ///
    /// * `range` - Element range
    ///
    /// ### Returns
    ///
    /// The elements converted to `T`, as `H5Dread` would return them.
    pub fn read<T: H5Elem>(&self, range: Range<usize>) -> Result<Vec<T>, ScDataErrors> {
        match &self.direct {
            Some(d) => d.read(range),
            None => Ok(self.ds.read_slice_1d::<T, _>(range)?.to_vec()),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Whether the linked HDF5 can report chunk locations at all.
    const CAN_INDEX: bool = cfg!(any(hdf5_chunk_info, hdf5_chunk_iter));

    /// Write `data` as a 1-D dataset.
    ///
    /// ### Params
    ///
    /// * `file` - Open file
    /// * `name` - Dataset name
    /// * `data` - Values
    /// * `chunk` - Chunk length, `None` for contiguous
    /// * `filters` - Filter pipeline, only applied when chunked
    ///
    /// ### Returns
    ///
    /// The dataset.
    fn write_ds<T: hdf5::H5Type>(
        file: &hdf5::File,
        name: &str,
        data: &[T],
        chunk: Option<usize>,
        filters: &[Filter],
    ) -> Dataset {
        let mut b = file.new_dataset_builder();
        if let Some(c) = chunk {
            b = b.chunk(c).set_filters(filters);
        }
        b.with_data(data).create(name).unwrap()
    }

    /// Assert that random ranges and the full range of the reader equal
    /// `H5Dread`.
    ///
    /// ### Params
    ///
    /// * `ds` - The dataset
    /// * `len` - Its length
    /// * `rng` - Range generator
    fn assert_matches_h5dread<T: H5Elem + PartialEq + std::fmt::Debug>(
        ds: Dataset,
        len: usize,
        rng: &mut StdRng,
    ) {
        let reader = H5ChunkedReader::new(ds.clone());
        for _ in 0..50 {
            let a = rng.random_range(0..=len);
            let b = rng.random_range(0..=len);
            let r = a.min(b)..a.max(b);
            let want: Vec<T> = ds.read_slice_1d::<T, _>(r.clone()).unwrap().to_vec();
            assert_eq!(reader.read::<T>(r.clone()).unwrap(), want, "range {r:?}");
        }
        let all: Vec<T> = ds.read_1d::<T>().unwrap().to_vec();
        assert_eq!(reader.read::<T>(0..len).unwrap(), all);
    }

    /// Create a scratch HDF5 file.
    ///
    /// ### Params
    ///
    /// * `name` - Unique suffix
    ///
    /// ### Returns
    ///
    /// The path, for removal, and the open file.
    fn temp_file(name: &str) -> (std::path::PathBuf, hdf5::File) {
        let path = std::env::temp_dir().join(format!("bixverse_h5_chunked_{name}.h5"));
        let file = hdf5::File::create(&path).unwrap();
        (path, file)
    }

    /// Both gzip pipelines, several stored and target types, chunk boundaries
    /// that never line up with the requested ranges.
    #[test]
    fn test_direct_pipelines_match_h5dread() {
        let (path, file) = temp_file("direct");
        let mut rng = StdRng::seed_from_u64(7);
        let len = 10_007;
        let ints: Vec<i64> = (0..len).map(|_| rng.random_range(0..40_000)).collect();
        let floats: Vec<f32> = (0..len).map(|_| rng.random_range(0..50) as f32).collect();
        let small: Vec<i32> = (0..len).map(|_| rng.random_range(-5..300)).collect();

        let pipelines = [
            vec![Filter::Deflate(4)],
            vec![Filter::Shuffle, Filter::Deflate(6)],
        ];
        for (p, filters) in pipelines.iter().enumerate() {
            let ds = write_ds(&file, &format!("i{p}"), &ints, Some(333), filters);
            assert_eq!(H5ChunkedReader::new(ds.clone()).is_direct(), CAN_INDEX);
            assert_matches_h5dread::<u32>(ds.clone(), len, &mut rng);
            assert_matches_h5dread::<usize>(ds.clone(), len, &mut rng);
            assert_matches_h5dread::<f32>(ds, len, &mut rng);

            let ds = write_ds(&file, &format!("f{p}"), &floats, Some(1000), filters);
            assert_eq!(H5ChunkedReader::new(ds.clone()).is_direct(), CAN_INDEX);
            assert_matches_h5dread::<f32>(ds.clone(), len, &mut rng);
            assert_matches_h5dread::<u32>(ds, len, &mut rng);

            // negatives clamp to zero, as in HDF5's hard conversion
            let ds = write_ds(&file, &format!("s{p}"), &small, Some(64), filters);
            assert_eq!(H5ChunkedReader::new(ds.clone()).is_direct(), CAN_INDEX);
            assert_matches_h5dread::<u32>(ds.clone(), len, &mut rng);
            assert_matches_h5dread::<f32>(ds, len, &mut rng);

            // single chunk larger than the dataset
            let ds = write_ds(&file, &format!("one{p}"), &floats[..50], Some(50), filters);
            assert_eq!(H5ChunkedReader::new(ds.clone()).is_direct(), CAN_INDEX);
            assert_matches_h5dread::<f32>(ds, 50, &mut rng);
        }
        drop(file);
        let _ = std::fs::remove_file(path);
    }

    /// Contiguous, unfiltered, other filters and partially written datasets
    /// take the `H5Dread` path and still read correctly.
    #[test]
    fn test_ineligible_datasets_fall_back() {
        let (path, file) = temp_file("fallback");
        let mut rng = StdRng::seed_from_u64(11);
        let len = 5_003;
        let ints: Vec<i64> = (0..len).map(|_| rng.random_range(0..40_000)).collect();

        let cases: [(&str, Option<usize>, Vec<Filter>); 4] = [
            ("contiguous", None, vec![]),
            ("chunked_raw", Some(500), vec![]),
            ("shuffle_only", Some(500), vec![Filter::Shuffle]),
            (
                "fletcher",
                Some(500),
                vec![Filter::Fletcher32, Filter::Deflate(4)],
            ),
        ];
        for (name, chunk, filters) in cases {
            let ds = write_ds(&file, name, &ints, chunk, &filters);
            assert!(!H5ChunkedReader::new(ds.clone()).is_direct(), "{name}");
            assert_matches_h5dread::<u32>(ds, len, &mut rng);
        }

        let mut b = file.new_dataset::<i64>();
        b = b.chunk(100).deflate(4);
        let ds = b.shape(1_000).create("unwritten").unwrap();
        ds.write_slice(&ints[..100], 0..100).unwrap();
        assert!(!H5ChunkedReader::new(ds.clone()).is_direct());
        assert_matches_h5dread::<u32>(ds, 1_000, &mut rng);

        drop(file);
        let _ = std::fs::remove_file(path);
    }

    /// Unshuffle inverts shuffle, including trailing bytes.
    #[test]
    fn test_unshuffle_round_trip() {
        let src: Vec<u8> = (0..23).collect();
        let size = 4;
        let n = src.len() / size;
        let mut shuffled = vec![0u8; src.len()];
        for i in 0..n {
            for j in 0..size {
                shuffled[j * n + i] = src[i * size + j];
            }
        }
        shuffled[n * size..].copy_from_slice(&src[n * size..]);
        let mut back = vec![0u8; src.len()];
        unshuffle(&shuffled, &mut back, size);
        assert_eq!(back, src);
    }
}
