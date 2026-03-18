//! Migration from v2 binary files to v3 format.
//!
//! v2 files have:
//! - CsrCellChunk: u16 raw counts, u16 gene indices, no discriminant byte
//! - CscGeneChunk: u16 raw counts, u32 cell indices, no discriminant byte
//!
//! v3 files have:
//! - CsrCellChunk: RawCounts enum (u16/u32), u32 gene indices, discriminant
//!   byte at header[29]
//! - CscGeneChunk: RawCounts enum (u16/u32), u32 cell indices, discriminant
//!   byte at header[33]

use bincode::{Decode, Encode, config, decode_from_slice};
use lz4_flex::decompress_size_prepended;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{
    CellGeneSparseWriter, CscGeneChunk, CsrCellChunk, RawCounts, SparseDataHeader,
};

/// Old File version
const V2_FILE_VERSION: u32 = 2;

/// v2 file header (same layout as v3, just a different version number)
#[repr(C)]
#[derive(Encode, Decode, Serialize, Deserialize)]
struct FileHeaderV2 {
    magic: [u8; 8],
    version: u32,
    main_header_offset: u64,
    cell_based: bool,
    _reserved_1: [u8; 32],
    _reserved_2: [u8; 3],
}

/// Read a v2 CsrCellChunk from a decompressed buffer.
///
/// v2 layout (header = 32 bytes):
///
/// ```text
/// [0..4)   data_raw_len    u32
/// [4..8)   data_norm_len   u32
/// [8..12)  col_indices_len u32
/// [12..20) library_size    u64
/// [20..28) original_index  u64
/// [28]     to_keep         u8
/// [29..32) padding         3 bytes (all zero)
/// ```
///
/// Followed by:
/// - data_raw: `data_raw_len * 2` bytes (u16)
/// - data_norm: `data_norm_len * 2` bytes (F16)
/// - indices: `col_indices_len * 2` bytes (u16!)
fn read_v2_cell_chunk(buffer: &[u8]) -> std::io::Result<CsrCellChunk> {
    if buffer.len() < 32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Buffer too small for v2 cell chunk header",
        ));
    }

    let header = &buffer[0..32];

    let data_raw_len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let data_norm_len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let col_indices_len =
        u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let library_size = u64::from_le_bytes([
        header[12], header[13], header[14], header[15], header[16], header[17], header[18],
        header[19],
    ]) as usize;
    let original_index = u64::from_le_bytes([
        header[20], header[21], header[22], header[23], header[24], header[25], header[26],
        header[27],
    ]) as usize;
    let to_keep = header[28] != 0;

    let data_start = 32;
    let data_end = data_start + data_raw_len * 2; // u16
    let norm_end = data_end + data_norm_len * 2; // F16

    // v2 raw counts: u16
    let data_raw_u16 = unsafe {
        let ptr = buffer.as_ptr().add(data_start) as *const u16;
        std::slice::from_raw_parts(ptr, data_raw_len).to_vec()
    };

    let data_norm: Vec<F16> = unsafe {
        let ptr = buffer.as_ptr().add(data_end) as *const u16;
        std::slice::from_raw_parts(ptr, data_norm_len)
            .iter()
            .map(|&bits| F16::from_bits(bits))
            .collect()
    };

    // v2 indices: u16 (the old code wrote indices with * 2)
    let indices_u16 = unsafe {
        let ptr = buffer.as_ptr().add(norm_end) as *const u16;
        std::slice::from_raw_parts(ptr, col_indices_len).to_vec()
    };

    // widen to u32 for v3
    let indices: Vec<u32> = indices_u16.iter().map(|&x| x as u32).collect();

    Ok(CsrCellChunk {
        data_raw: RawCounts::U16(data_raw_u16),
        data_norm,
        library_size,
        indices,
        original_index,
        to_keep,
    })
}

/// Read a v2 CscGeneChunk from a decompressed buffer.
///
/// v2 layout (header = 36 bytes):
///
/// ```text
/// [0..4)   data_raw_len    u32
/// [4..8)   data_norm_len   u32
/// [8..12)  indices_len     u32
/// [12..14) avg_exp         F16
/// [14..16) padding         2 bytes
/// [16..24) nnz             u64
/// [24..32) original_index  u64
/// [32]     to_keep         u8
/// [33..36) padding         3 bytes (all zero)
/// ```
///
/// Followed by:
/// - data_raw: `data_raw_len * 2` bytes (u16)
/// - data_norm: `data_norm_len * 2` bytes (F16)
/// - indices: `indices_len * 4` bytes (u32 -- gene chunks always had u32 cell indices)
fn read_v2_gene_chunk(buffer: &[u8]) -> std::io::Result<CscGeneChunk> {
    if buffer.len() < 36 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Buffer too small for v2 gene chunk header",
        ));
    }

    let header = &buffer[0..36];

    let data_raw_len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let data_norm_len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let row_indices_len =
        u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let avg_exp = F16::from_le_bytes([header[12], header[13]]);
    // skip 2 bytes padding
    let nnz = u64::from_le_bytes([
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ]) as usize;
    let original_index = u64::from_le_bytes([
        header[24], header[25], header[26], header[27], header[28], header[29], header[30],
        header[31],
    ]) as usize;
    let to_keep = header[32] != 0;

    let data_start = 36;
    let data_end = data_start + data_raw_len * 2; // u16
    let norm_end = data_end + data_norm_len * 2; // F16

    // v2 raw counts: u16
    let data_raw_u16 = unsafe {
        let ptr = buffer.as_ptr().add(data_start) as *const u16;
        std::slice::from_raw_parts(ptr, data_raw_len).to_vec()
    };

    let data_norm: Vec<F16> = unsafe {
        let ptr = buffer.as_ptr().add(data_end) as *const u16;
        let slice = std::slice::from_raw_parts(ptr, data_norm_len);
        slice.iter().map(|&bits| F16::from_bits(bits)).collect()
    };

    // v2 gene chunk indices were always u32
    let indices = unsafe {
        let ptr = buffer.as_ptr().add(norm_end) as *const u32;
        std::slice::from_raw_parts(ptr, row_indices_len).to_vec()
    };

    Ok(CscGeneChunk {
        data_raw: RawCounts::U16(data_raw_u16),
        data_norm,
        avg_exp,
        nnz,
        indices,
        original_index,
        to_keep,
    })
}

/// Migrate a v2 binary file to v3 format.
///
/// Reads the entire v2 file, re-encodes every chunk in the v3 format
/// (RawCounts discriminant byte, u32 gene indices for cell chunks), and
/// writes a new file. The original file is not modified.
///
/// ### Params
///
/// * `input_path` - Path to the v2 binary file.
/// * `output_path` - Path to write the v3 binary file.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Ok(()) on success, or an IO error.
pub fn migrate_v2_to_v3<P: AsRef<Path>>(
    input_path: P,
    output_path: P,
    verbose: bool,
) -> std::io::Result<()> {
    let file = File::open(input_path.as_ref())?;
    let file_size = file.metadata()?.len();

    let mmap = unsafe {
        let mut opts = MmapOptions::new();
        if file_size <= 8 * 1024 * 1024 * 1024 {
            opts.populate();
        }
        opts.map(&file)?
    };

    // parse v2 file header
    let file_header_bytes = &mmap[0..64];
    let (file_header, _) =
        decode_from_slice::<FileHeaderV2, _>(file_header_bytes, config::standard()).map_err(
            |_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to decode v2 file header",
                )
            },
        )?;

    if file_header.version != V2_FILE_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Expected file version {}, got {}",
                V2_FILE_VERSION, file_header.version
            ),
        ));
    }

    let cell_based = file_header.cell_based;

    // read the main header
    let main_header_offset = file_header.main_header_offset as usize;
    let header_size = u64::from_le_bytes(
        mmap[main_header_offset..main_header_offset + 8]
            .try_into()
            .unwrap(),
    ) as usize;

    let header_bytes = &mmap[main_header_offset + 8..main_header_offset + 8 + header_size];
    let (header, _) = decode_from_slice::<SparseDataHeader, _>(header_bytes, config::standard())
        .map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to decode header")
        })?;

    let chunks_start: u64 = 64;
    let total_chunks = header.no_chunks;

    if verbose {
        println!(
            "Migrating v2 -> v3: {} ({} chunks, cell_based={})",
            input_path.as_ref().display(),
            total_chunks.separate_with_underscores(),
            cell_based,
        );
    }

    // set up the v3 writer
    let mut writer = CellGeneSparseWriter::new(
        output_path.as_ref(),
        cell_based,
        header.total_cells,
        header.total_genes,
    )?;

    // build reverse map: chunk_position -> original_index
    let mut chunk_to_original: Vec<(usize, usize)> = header
        .index_map
        .iter()
        .map(|(&original_idx, &chunk_idx)| (chunk_idx, original_idx))
        .collect();
    chunk_to_original.sort_by_key(|&(chunk_idx, _)| chunk_idx);

    for (chunk_pos, (_, _original_idx)) in chunk_to_original.iter().enumerate() {
        let chunk_offset = (chunks_start + header.chunk_offsets[chunk_pos]) as usize;

        let compressed_size =
            u64::from_le_bytes(mmap[chunk_offset..chunk_offset + 8].try_into().unwrap()) as usize;

        let compressed = &mmap[chunk_offset + 8..chunk_offset + 8 + compressed_size];
        let decompressed = decompress_size_prepended(compressed).unwrap();

        if cell_based {
            let chunk = read_v2_cell_chunk(&decompressed)?;
            writer.write_cell_chunk(chunk)?;
        } else {
            let chunk = read_v2_gene_chunk(&decompressed)?;
            writer.write_gene_chunk(chunk)?;
        }

        if verbose && (chunk_pos + 1) % 100_000 == 0 {
            println!(
                "  Migrated {} / {} chunks",
                (chunk_pos + 1).separate_with_underscores(),
                total_chunks.separate_with_underscores(),
            );
        }
    }

    writer.finalise()?;

    if verbose {
        println!(
            "  Migrated {} / {} chunks (complete). Written to: {}",
            total_chunks.separate_with_underscores(),
            total_chunks.separate_with_underscores(),
            output_path.as_ref().display(),
        );
    }

    Ok(())
}

/// Migrate a matched pair of v2 cell and gene binary files to v3.
///
/// Convenience wrapper that calls `migrate_v2_to_v3` for both the
/// cell-based and gene-based files.
///
/// ### Params
///
/// * `cell_input` - Path to the v2 cell-based binary file.
/// * `cell_output` - Path to write the v3 cell-based binary file.
/// * `gene_input` - Path to the v2 gene-based binary file.
/// * `gene_output` - Path to write the v3 gene-based binary file.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Ok(()) on success, or an IO error.
pub fn migrate_v2_to_v3_pair<P: AsRef<Path>>(
    cell_input: P,
    cell_output: P,
    gene_input: P,
    gene_output: P,
    verbose: bool,
) -> std::io::Result<()> {
    if verbose {
        println!("Migrating cell-based file...");
    }
    migrate_v2_to_v3(&cell_input, &cell_output, verbose)?;

    if verbose {
        println!("Migrating gene-based file...");
    }
    migrate_v2_to_v3(&gene_input, &gene_output, verbose)?;

    if verbose {
        println!("Migration complete.");
    }

    Ok(())
}
