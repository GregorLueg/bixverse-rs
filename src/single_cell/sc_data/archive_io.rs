//! Cold-storage archive of the cell-based binary.
//!
//! The two on-disk views are laid out for random access, not size: every
//! non-zero is stored twice (CSR and CSC), each copy carries a derivable f16
//! norm, and lz4 only sees one chunk at a time. The archive keeps the cell
//! file only, drops every norm that recomputes bit for bit from the raw
//! counts, splits blocks of cells into per-field streams (gene index deltas,
//! raw counts, per-cell scalars), byte-shuffles them and compresses each block
//! as an independent zstd frame. Restore rebuilds the cell file through the
//! regular writer and the gene file via [`write_gene_file`].
//!
//! Layout:
//!
//! ```text
//! [0..8)    magic          b"BXARCHV1"
//! [8..12)   version        u32
//! [12..16)  target_size    f32
//! [16..24)  total_cells    u64
//! [24..32)  total_genes    u64
//! [32..40)  n_chunks       u64
//! then per block: frame_len u64, zstd frame (content checksum on)
//! ```

use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    time::Instant,
};
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::*;
use crate::single_cell::sc_data::gene_file_io::write_gene_file;

////////////
// Consts //
////////////

/// Leading bytes identifying a bixverse count archive.
const ARCHIVE_MAGIC: &[u8; 8] = b"BXARCHV1";

/// Archive format version. Independent of `SC_FILE_VERSION`.
const ARCHIVE_VERSION: u32 = 1;

/// Fixed header length, see the module docs.
const ARCHIVE_HEADER_LEN: usize = 40;

/// Cells per zstd frame.
///
/// One block per worker is decoded at a time, so peak memory is roughly
/// `n_threads * ARCHIVE_BLOCK_CELLS * mean_nnz_per_cell * 24 B` (decoded chunk
/// plus the stream buffer). Larger blocks buy little: zstd windows top out at
/// a few MB below level 20, which a 2,048-cell block already exceeds.
const ARCHIVE_BLOCK_CELLS: usize = 2_048;

//////////////////
// ArchiveStats //
//////////////////

/// Summary of a written archive.
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Cells written.
    pub n_cells: usize,
    /// Non-zeros written.
    pub nnz: usize,
    /// Cells whose norm did not recompute and was stored verbatim.
    pub n_norm_stored: usize,
    /// Size of the archive on disk in bytes.
    pub archive_bytes: u64,
}

/////////////
// Helpers //
/////////////

/// Byte-shuffle fixed-width little-endian values: byte `b` of every value is
/// grouped together, so the near-constant high bytes form long runs.
///
/// ### Params
///
/// * `bytes` - Concatenated values
/// * `width` - Bytes per value
///
/// ### Returns
///
/// The shuffled bytes.
fn shuffle(bytes: &[u8], width: usize) -> Vec<u8> {
    let n = bytes.len() / width;
    let mut out = vec![0u8; bytes.len()];
    for (i, value) in bytes.chunks_exact(width).enumerate() {
        for (b, &byte) in value.iter().enumerate() {
            out[b * n + i] = byte;
        }
    }
    out
}

/// Inverse of [`shuffle`].
///
/// ### Params
///
/// * `bytes` - Shuffled values
/// * `width` - Bytes per value
///
/// ### Returns
///
/// The values in their original byte order.
fn unshuffle(bytes: &[u8], width: usize) -> Vec<u8> {
    let n = bytes.len() / width;
    let mut out = vec![0u8; bytes.len()];
    for (i, value) in out.chunks_exact_mut(width).enumerate() {
        for (b, byte) in value.iter_mut().enumerate() {
            *byte = bytes[b * n + i];
        }
    }
    out
}

////////////
// Cursor //
////////////

/// Bounds-checked cursor over a decompressed block.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Take the next `n` bytes.
    ///
    /// ### Params
    ///
    /// * `n` - Number of bytes
    ///
    /// ### Returns
    ///
    /// The slice, or [`BixverseErrors::ArchiveCorrupt`] past the end.
    fn take(&mut self, n: usize) -> Result<&'a [u8], BixverseErrors> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| BixverseErrors::ArchiveCorrupt("block truncated".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Take `count` shuffled values of `width` bytes and unshuffle them.
    fn take_shuffled(&mut self, count: usize, width: usize) -> Result<Vec<u8>, BixverseErrors> {
        let n = count
            .checked_mul(width)
            .ok_or_else(|| BixverseErrors::ArchiveCorrupt("length overflow".into()))?;
        Ok(unshuffle(self.take(n)?, width))
    }
}

/// Little-endian `u32`s from a byte slice of exact length.
fn le_u32(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunk by construction")))
}

/// Little-endian `u64`s from a byte slice of exact length.
fn le_u64(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8-byte chunk by construction")))
}

/// Does the stored norm of a cell recompute bit for bit from its raw counts?
///
/// ### Params
///
/// * `cell` - The cell
/// * `target_size` - Target size from the file header, `0.0` if unknown
///
/// ### Returns
///
/// `true` if the norm can be dropped from the archive.
fn norm_derivable(cell: &CsrCellChunk, target_size: f32) -> bool {
    if target_size == 0.0 {
        return false;
    }
    let lib = cell.library_size as f32;
    cell.data_raw
        .iter()
        .zip(&cell.data_norm)
        .all(|(x, n)| norm_value(x as f32, lib, target_size).to_bits() == n.to_bits())
}

//////////////////////
// Block (de)coding //
//////////////////////

/// Serialise a block of cells into per-field, byte-shuffled streams.
///
/// ```text
/// n_cells u32
/// nnz u32, library_size u64, original_index delta u64 (all shuffled)
/// raw_elem_size u8, to_keep u8, norm_stored u8
/// gene index deltas u32, raw counts u32 (shuffled, length total nnz)
/// stored norms u16 (shuffled, only for cells with norm_stored = 1)
/// ```
///
/// ### Params
///
/// * `cells` - The cells, in file order
/// * `target_size` - Target size from the file header
///
/// ### Returns
///
/// `(stream, nnz, n_norm_stored)`.
fn encode_block(
    cells: &[CsrCellChunk],
    target_size: f32,
) -> Result<(Vec<u8>, usize, usize), BixverseErrors> {
    let n = cells.len();
    let total_nnz: usize = cells.iter().map(|c| c.indices.len()).sum();

    let mut nnz = Vec::with_capacity(n * 4);
    let mut lib = Vec::with_capacity(n * 8);
    let mut orig = Vec::with_capacity(n * 8);
    let mut elem = Vec::with_capacity(n);
    let mut keep = Vec::with_capacity(n);
    let mut norm_flag = Vec::with_capacity(n);
    let mut idx = Vec::with_capacity(total_nnz * 4);
    let mut raw = Vec::with_capacity(total_nnz * 4);
    let mut norm = Vec::new();

    let mut prev_orig = 0u64;
    for cell in cells {
        let len = cell.indices.len();
        if cell.data_raw.len() != len || cell.data_norm.len() != len {
            return Err(BixverseErrors::ArchiveCorrupt(format!(
                "cell {} has layers of unequal length",
                cell.original_index
            )));
        }

        nnz.extend_from_slice(&(len as u32).to_le_bytes());
        lib.extend_from_slice(&(cell.library_size as u64).to_le_bytes());
        let o = cell.original_index as u64;
        orig.extend_from_slice(&o.wrapping_sub(prev_orig).to_le_bytes());
        prev_orig = o;
        elem.push(cell.data_raw.elem_size());
        keep.push(cell.to_keep as u8);

        // wrapping deltas stay lossless if a row is ever unsorted
        let mut prev = 0u32;
        for &g in &cell.indices {
            idx.extend_from_slice(&g.wrapping_sub(prev).to_le_bytes());
            prev = g;
        }
        for x in cell.data_raw.iter() {
            raw.extend_from_slice(&x.to_le_bytes());
        }

        if norm_derivable(cell, target_size) {
            norm_flag.push(0);
        } else {
            norm_flag.push(1);
            for v in &cell.data_norm {
                norm.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    let n_norm_stored = norm_flag.iter().filter(|&&f| f == 1).count();

    let mut out = Vec::with_capacity(4 + n * 23 + total_nnz * 8 + norm.len());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend(shuffle(&nnz, 4));
    out.extend(shuffle(&lib, 8));
    out.extend(shuffle(&orig, 8));
    out.extend(elem);
    out.extend(keep);
    out.extend(norm_flag);
    out.extend(shuffle(&idx, 4));
    out.extend(shuffle(&raw, 4));
    out.extend(shuffle(&norm, 2));

    Ok((out, total_nnz, n_norm_stored))
}

/// Inverse of [`encode_block`].
///
/// ### Params
///
/// * `buf` - Decompressed block stream
/// * `target_size` - Target size from the archive header
///
/// ### Returns
///
/// The cells, in file order.
fn decode_block(buf: &[u8], target_size: f32) -> Result<Vec<CsrCellChunk>, BixverseErrors> {
    let mut cur = Cursor { buf, pos: 0 };
    let n = u32::from_le_bytes(cur.take(4)?.try_into().expect("4 bytes")) as usize;

    let nnz: Vec<usize> = le_u32(&cur.take_shuffled(n, 4)?)
        .map(|v| v as usize)
        .collect();
    let lib: Vec<u64> = le_u64(&cur.take_shuffled(n, 8)?).collect();
    let orig_delta: Vec<u64> = le_u64(&cur.take_shuffled(n, 8)?).collect();
    let elem = cur.take(n)?;
    let keep = cur.take(n)?;
    let norm_flag = cur.take(n)?;

    let total_nnz: usize = nnz.iter().sum();
    let n_norm: usize = nnz
        .iter()
        .zip(norm_flag)
        .filter(|&(_, &f)| f == 1)
        .map(|(&k, _)| k)
        .sum();

    let idx: Vec<u32> = le_u32(&cur.take_shuffled(total_nnz, 4)?).collect();
    let raw: Vec<u32> = le_u32(&cur.take_shuffled(total_nnz, 4)?).collect();
    let norm_bytes = cur.take_shuffled(n_norm, 2)?;
    let mut stored_norm = norm_bytes
        .chunks_exact(2)
        .map(|c| F16::from_le_bytes(c.try_into().expect("2-byte chunk by construction")));

    let mut cells = Vec::with_capacity(n);
    let (mut start, mut prev_orig) = (0usize, 0u64);
    for i in 0..n {
        let end = start + nnz[i];
        let raw_i = &raw[start..end];

        let mut prev = 0u32;
        let indices: Vec<u32> = idx[start..end]
            .iter()
            .map(|&d| {
                prev = prev.wrapping_add(d);
                prev
            })
            .collect();

        let data_raw = match elem[i] {
            RAW_ELEM_U32 => RawCounts::U32(raw_i.to_vec()),
            RAW_ELEM_U16 => RawCounts::U16(
                raw_i
                    .iter()
                    .map(|&x| u16::try_from(x))
                    .collect::<Result<_, _>>()
                    .map_err(|_| BixverseErrors::ArchiveCorrupt("u16 count overflow".into()))?,
            ),
            other => return Err(BixverseErrors::RawElemSizeInvalid(other)),
        };

        let data_norm: Vec<F16> = match norm_flag[i] {
            0 => {
                let l = lib[i] as f32;
                raw_i
                    .iter()
                    .map(|&x| norm_value(x as f32, l, target_size))
                    .collect()
            }
            1 => stored_norm.by_ref().take(nnz[i]).collect(),
            _ => return Err(BixverseErrors::ArchiveCorrupt("invalid norm flag".into())),
        };

        prev_orig = prev_orig.wrapping_add(orig_delta[i]);
        cells.push(CsrCellChunk {
            data_raw,
            data_norm,
            library_size: lib[i] as usize,
            indices,
            original_index: prev_orig as usize,
            to_keep: keep[i] != 0,
        });
        start = end;
    }

    Ok(cells)
}

/// Compress a block stream into a checksummed zstd frame.
///
/// ### Params
///
/// * `raw` - Block stream from [`encode_block`]
/// * `level` - zstd level
///
/// ### Returns
///
/// The frame.
fn compress_frame(raw: &[u8], level: i32) -> std::io::Result<Vec<u8>> {
    let mut enc = zstd::stream::write::Encoder::new(Vec::with_capacity(raw.len() / 4), level)?;
    enc.include_checksum(true)?;
    enc.write_all(raw)?;
    enc.finish()
}

/////////////
// Archive //
/////////////

/// Archive a cell-based binary.
///
/// The gene-based binary is not archived; [`restore_archive`] rebuilds it.
/// Blocks of [`ARCHIVE_BLOCK_CELLS`] cells are read, encoded and compressed in
/// parallel, one batch of `n_threads` blocks at a time, and appended in file
/// order.
///
/// ### Params
///
/// * `cell_path` - Path to `counts_cells.bin`
/// * `out_path` - Path of the archive to write
/// * `level` - zstd compression level (1 to 22)
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The [`ArchiveStats`].
pub fn archive_cell_file(
    cell_path: &str,
    out_path: &str,
    level: i32,
    verbose: bool,
) -> Result<ArchiveStats, BixverseErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::new(cell_path)?;
    if !reader.is_cell_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }
    let header = reader.get_header();
    let target_size = reader.target_size().unwrap_or(0.0);

    // file order, so the restored file lays chunks out identically
    let mut order = vec![usize::MAX; header.no_chunks];
    for (&orig, &chunk) in &header.index_map {
        *order.get_mut(chunk).ok_or_else(|| {
            BixverseErrors::ArchiveCorrupt(format!("chunk {chunk} outside the file"))
        })? = orig;
    }

    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, File::create(out_path)?);
    writer.write_all(ARCHIVE_MAGIC)?;
    writer.write_all(&ARCHIVE_VERSION.to_le_bytes())?;
    writer.write_all(&target_size.to_le_bytes())?;
    writer.write_all(&(header.total_cells as u64).to_le_bytes())?;
    writer.write_all(&(header.total_genes as u64).to_le_bytes())?;
    writer.write_all(&(header.no_chunks as u64).to_le_bytes())?;

    let blocks: Vec<&[usize]> = order.chunks(ARCHIVE_BLOCK_CELLS).collect();
    let (mut nnz, mut n_norm_stored) = (0usize, 0usize);

    for batch in blocks.chunks(rayon::current_num_threads().max(1)) {
        let frames = batch
            .par_iter()
            .map(|indices| {
                let cells = reader.read_cells_parallel(indices)?;
                let (stream, k, s) = encode_block(&cells, target_size)?;
                drop(cells);
                Ok((compress_frame(&stream, level)?, k, s))
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        for (frame, k, s) in frames {
            writer.write_all(&(frame.len() as u64).to_le_bytes())?;
            writer.write_all(&frame)?;
            nnz += k;
            n_norm_stored += s;
        }
    }
    writer.flush()?;
    drop(writer);

    let stats = ArchiveStats {
        n_cells: header.no_chunks,
        nnz,
        n_norm_stored,
        archive_bytes: std::fs::metadata(out_path)?.len(),
    };

    if verbose {
        println!(
            "  Archived {} cells / {} non-zeros into {} bytes ({:.3} B/nnz, {} cells with stored norm) in {:.2?}.",
            stats.n_cells.separate_with_underscores(),
            stats.nnz.separate_with_underscores(),
            stats.archive_bytes.separate_with_underscores(),
            stats.archive_bytes as f64 / stats.nnz.max(1) as f64,
            stats.n_norm_stored.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(stats)
}

/////////////
// Restore //
/////////////

/// Restore both binaries from an archive.
///
/// Frames are decompressed and decoded in parallel batches, re-serialised
/// through the regular lz4 chunk path and appended in archive order, so the
/// cell file comes back with the original chunk layout. The gene file is then
/// built from it with [`write_gene_file`].
///
/// ### Params
///
/// * `archive_path` - Path to the archive
/// * `cell_path` - Path of `counts_cells.bin` to write
/// * `gene_path` - Path of `counts_genes.bin` to write
/// * `max_nnz_in_memory` - Passed to [`write_gene_file`]; `None` converts in
///   a single phase.
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once both files are finalised.
pub fn restore_archive(
    archive_path: &str,
    cell_path: &str,
    gene_path: &str,
    max_nnz_in_memory: Option<usize>,
    verbose: bool,
) -> Result<(), BixverseErrors> {
    let start = Instant::now();
    let file = File::open(archive_path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(64 * 1024 * 1024, file);

    if file_len < ARCHIVE_HEADER_LEN as u64 {
        return Err(BixverseErrors::ArchiveMagicMismatch);
    }
    let mut head = [0u8; ARCHIVE_HEADER_LEN];
    reader.read_exact(&mut head)?;
    if &head[0..8] != ARCHIVE_MAGIC {
        return Err(BixverseErrors::ArchiveMagicMismatch);
    }
    let version = u32::from_le_bytes(head[8..12].try_into().expect("4 bytes"));
    if version != ARCHIVE_VERSION {
        return Err(BixverseErrors::ArchiveVersionMismatch {
            expected: ARCHIVE_VERSION,
            found: version,
        });
    }
    let target_size = f32::from_le_bytes(head[12..16].try_into().expect("4 bytes"));
    let total_cells = u64::from_le_bytes(head[16..24].try_into().expect("8 bytes")) as usize;
    let total_genes = u64::from_le_bytes(head[24..32].try_into().expect("8 bytes")) as usize;
    let n_chunks = u64::from_le_bytes(head[32..40].try_into().expect("8 bytes")) as usize;

    let mut writer =
        CellGeneSparseWriter::new(cell_path, true, total_cells, total_genes, target_size)?;

    let batch_size = rayon::current_num_threads().max(1);
    let mut pos = ARCHIVE_HEADER_LEN as u64;
    let mut written = 0usize;

    while pos < file_len {
        let mut frames = Vec::with_capacity(batch_size);
        while frames.len() < batch_size && pos < file_len {
            if file_len - pos < 8 {
                return Err(BixverseErrors::ArchiveCorrupt(format!(
                    "frame header at byte {pos} truncated"
                )));
            }
            let mut len_bytes = [0u8; 8];
            reader.read_exact(&mut len_bytes)?;
            let len = u64::from_le_bytes(len_bytes);
            if len > file_len - pos - 8 {
                return Err(BixverseErrors::ArchiveCorrupt(format!(
                    "frame at byte {pos} runs past the end of the file"
                )));
            }
            let mut frame = vec![0u8; len as usize];
            reader.read_exact(&mut frame)?;
            pos += 8 + len;
            frames.push(frame);
        }

        let decoded = frames
            .par_iter()
            .map(|frame| {
                let stream = zstd::stream::decode_all(&frame[..])
                    .map_err(|e| BixverseErrors::ArchiveCorrupt(e.to_string()))?;
                decode_block(&stream, target_size)?
                    .into_iter()
                    .map(|c| Ok((c.original_index, c.to_compressed_bytes()?)))
                    .collect::<Result<Vec<_>, BixverseErrors>>()
            })
            .collect::<Result<Vec<_>, BixverseErrors>>()?;

        for payloads in decoded {
            written += payloads.len();
            writer.write_compressed_cell_chunks(&payloads)?;
        }
    }

    if written != n_chunks {
        return Err(BixverseErrors::ArchiveCorrupt(format!(
            "expected {n_chunks} cells, found {written}"
        )));
    }
    writer.finalise()?;

    if verbose {
        println!(
            "  Restored {} cells in {:.2?}; building the gene file.",
            written.separate_with_underscores(),
            start.elapsed()
        );
    }

    write_gene_file(cell_path, gene_path, max_nnz_in_memory, verbose)?;

    if verbose {
        println!("  Restore finished in {:.2?}.", start.elapsed());
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard that removes a test's temp file even if an assert fails.
    struct TempBin(std::path::PathBuf);

    /// Errors are ignored: the file may already be gone, and this runs
    /// during unwind.
    impl Drop for TempBin {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempBin {
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_archive_io_{name}.bin")))
        }

        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Cell `i` expresses gene `g` when `(i * 7 + g * 3) % 5 == 0`, count
    /// `1 + (i + g) % 4`; cell 2 is empty. Cells in `odd_norm` get a norm that
    /// does not recompute, and cell 5 gets a count above `u16::MAX`.
    fn write_cells(path: &str, n_cells: usize, n_genes: usize, target: f32, odd_norm: &[usize]) {
        let mut writer = CellGeneSparseWriter::new(path, true, n_cells, n_genes, target).unwrap();
        for i in 0..n_cells {
            let (mut data, mut idx) = (Vec::new(), Vec::new());
            if i != 2 {
                for g in 0..n_genes {
                    if (i * 7 + g * 3) % 5 == 0 {
                        data.push(if i == 5 {
                            70_000
                        } else {
                            1 + ((i + g) % 4) as u32
                        });
                        idx.push(g as u32);
                    }
                }
            }
            let mut chunk = CsrCellChunk::from_data(&data, &idx, i, target.max(1.0), true);
            if odd_norm.contains(&i) {
                chunk
                    .data_norm
                    .iter_mut()
                    .for_each(|v| *v = F16::from_f32(1.5));
            }
            writer.write_cell_chunk(chunk).unwrap();
        }
        writer.finalise().unwrap();
    }

    #[allow(clippy::type_complexity)]
    fn read_cells(path: &str) -> Vec<(Vec<u32>, Vec<u16>, Vec<u32>, usize, usize, bool, u8)> {
        ParallelSparseReader::new(path)
            .unwrap()
            .get_all_cells()
            .unwrap()
            .into_iter()
            .map(|c| {
                (
                    c.data_raw.iter().collect(),
                    c.data_norm.iter().map(|v| v.to_bits()).collect(),
                    c.indices,
                    c.library_size,
                    c.original_index,
                    c.to_keep,
                    c.data_raw.elem_size(),
                )
            })
            .collect()
    }

    #[test]
    fn test_shuffle_round_trip() {
        let bytes: Vec<u8> = (0..48u8).collect();
        for w in [1, 2, 4, 8] {
            assert_eq!(unshuffle(&shuffle(&bytes, w), w), bytes);
        }
    }

    #[test]
    fn test_archive_round_trip_across_blocks() {
        let cells = TempBin::new("rt_cells");
        let genes = TempBin::new("rt_genes");
        let archive = TempBin::new("rt_archive");
        let cells_out = TempBin::new("rt_cells_out");
        let genes_out = TempBin::new("rt_genes_out");

        let n_cells = 2 * ARCHIVE_BLOCK_CELLS + 5;
        write_cells(
            cells.path(),
            n_cells,
            11,
            1e4,
            &[7, ARCHIVE_BLOCK_CELLS + 1],
        );
        write_gene_file(cells.path(), genes.path(), None, false).unwrap();

        let stats = archive_cell_file(cells.path(), archive.path(), 3, false).unwrap();
        assert_eq!(stats.n_cells, n_cells);
        assert_eq!(stats.n_norm_stored, 2);

        restore_archive(
            archive.path(),
            cells_out.path(),
            genes_out.path(),
            None,
            false,
        )
        .unwrap();

        assert_eq!(read_cells(cells_out.path()), read_cells(cells.path()));
        assert_eq!(
            std::fs::read(cells_out.path()).unwrap(),
            std::fs::read(cells.path()).unwrap()
        );
        assert_eq!(
            std::fs::read(genes_out.path()).unwrap(),
            std::fs::read(genes.path()).unwrap()
        );
    }

    #[test]
    fn test_archive_unknown_target_size_stores_norm() {
        let cells = TempBin::new("raw_cells");
        let archive = TempBin::new("raw_archive");
        let cells_out = TempBin::new("raw_cells_out");
        let genes_out = TempBin::new("raw_genes_out");

        write_cells(cells.path(), 40, 9, 0.0, &[]);

        let stats = archive_cell_file(cells.path(), archive.path(), 3, false).unwrap();
        assert_eq!(stats.n_norm_stored, 40);

        restore_archive(
            archive.path(),
            cells_out.path(),
            genes_out.path(),
            None,
            false,
        )
        .unwrap();
        assert_eq!(read_cells(cells_out.path()), read_cells(cells.path()));
    }

    #[test]
    fn test_restore_rejects_corrupt_archive() {
        let cells = TempBin::new("bad_cells");
        let archive = TempBin::new("bad_archive");
        let cells_out = TempBin::new("bad_cells_out");
        let genes_out = TempBin::new("bad_genes_out");

        write_cells(cells.path(), 40, 9, 1e4, &[]);
        archive_cell_file(cells.path(), archive.path(), 3, false).unwrap();
        let bytes = std::fs::read(archive.path()).unwrap();
        let restore = || {
            restore_archive(
                archive.path(),
                cells_out.path(),
                genes_out.path(),
                None,
                false,
            )
        };

        std::fs::write(archive.path(), &bytes[..bytes.len() - 3]).unwrap();
        assert!(matches!(restore(), Err(BixverseErrors::ArchiveCorrupt(_))));

        let mut flipped = bytes.clone();
        let mid = ARCHIVE_HEADER_LEN + 8 + (bytes.len() - ARCHIVE_HEADER_LEN - 8) / 2;
        flipped[mid] ^= 0xff;
        std::fs::write(archive.path(), &flipped).unwrap();
        assert!(restore().is_err());

        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        std::fs::write(archive.path(), &wrong).unwrap();
        assert!(matches!(
            restore(),
            Err(BixverseErrors::ArchiveMagicMismatch)
        ));
    }
}
