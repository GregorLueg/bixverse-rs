//! Converters from the v3 and v4 layouts to v5, and the in-place upgrade of
//! a store.

use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::store::chunk::norm_value;
use crate::store::v4::{ChunkCodec, V4Header, snap_v3_norm};
use crate::store::writer::{CellGeneSparseWriter, CellPayload};

////////////
// Consts //
////////////

/// Cell chunks converted per parallel batch. Bounds the decoded v3 chunks and
/// the compressed v4 payloads held at once.
const CONVERT_CELL_BATCH: usize = 4_096;

/// Gene chunks converted per parallel batch. Genes are far larger than cells
/// (one entry per expressing cell), so the batch is much smaller.
const CONVERT_GENE_BATCH: usize = 64;

////////////////
// Converters //
////////////////

/// Refuse anything but a v3 or v4 input.
///
/// ### Params
///
/// * `reader` - The opened input
///
/// ### Returns
///
/// `Ok(())` for v3 and v4, [`ScDataErrors::FileVersionMismatch`] otherwise.
fn require_pre_v5(reader: &ParallelSparseReader) -> Result<(), ScDataErrors> {
    match reader.file_version() {
        SC_FILE_VERSION_V3 | SC_FILE_VERSION_V4 => Ok(()),
        found => Err(ScDataErrors::FileVersionMismatch {
            expected: SC_FILE_VERSION_V4,
            found,
        }),
    }
}

/// Convert a v3 or v4 cell-based file to v5.
///
/// Streams in batches of [`CONVERT_CELL_BATCH`] chunks, in file order, so the
/// original index of every chunk is kept. A v3 norm that is the f16 rounding
/// of the recompute is dropped (see [`snap_v3_norm`]); a verbatim v4 norm
/// widens to f32 exactly.
///
/// ### Params
///
/// * `cell_in` - Path to the v3 or v4 cell-based file
/// * `cell_out` - Path of the v5 file to write
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Library sizes indexed by original cell index, `total_cells` long at least,
/// which [`convert_gene_file_to_v5`] needs to drop the gene-file norm.
pub fn convert_cell_file_to_v5(
    cell_in: &str,
    cell_out: &str,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<Vec<u64>, ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::open_any(cell_in)?;
    require_pre_v5(&reader)?;
    if !reader.is_cell_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "gene-based",
            requested: "cell-based",
        });
    }
    let dims = reader.dims();
    let from_v3 = reader.file_version() == SC_FILE_VERSION_V3;
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = reader.chunk_order()?;

    let writer = CellGeneSparseWriter::new(
        cell_out,
        true,
        dims.total_cells,
        dims.total_genes,
        target_size,
    )?;
    let mut writer = writer.with_codec(codec);
    let format = writer.format();

    let table_len = dims
        .total_cells
        .max(order.iter().copied().max().map_or(0, |m| m + 1));
    let mut lib_table = vec![0u64; table_len];
    let mut stored_norm = 0usize;

    for batch in order.chunks(CONVERT_CELL_BATCH) {
        let cells = reader.read_cells_parallel(batch)?;
        let payloads = cells
            .into_par_iter()
            .map(|mut cell| {
                if from_v3 {
                    let lib = cell.library_size as f32;
                    let raw: Vec<u32> = cell.data_raw.iter().collect();
                    snap_v3_norm(&mut cell.data_norm, |k| {
                        norm_value(raw[k] as f32, lib, target_size)
                    });
                }
                Ok(CellPayload {
                    original_index: cell.original_index,
                    library_size: cell.library_size,
                    bytes: cell.to_v4_bytes(&format)?,
                })
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for p in &payloads {
            lib_table[p.original_index] = p.library_size as u64;
            stored_norm += V4Header::parse(&p.bytes)?.norm_stored() as usize;
        }
        writer.write_compressed_cell_chunks(&payloads)?;
    }
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} cells to v5 ({} with stored norm) in {:.2?}.",
            order.len().separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(lib_table)
}

/// Convert a v3 or v4 gene-based file to v5.
///
/// Streams in batches of [`CONVERT_GENE_BATCH`] chunks, in file order. The
/// output carries `cell_lib` as its library-size table.
///
/// ### Params
///
/// * `gene_in` - Path to the v3 or v4 gene-based file
/// * `gene_out` - Path of the v5 file to write
/// * `cell_lib` - Library sizes indexed by original cell index, as returned by
///   [`convert_cell_file_to_v5`]
/// * `codec` - Body codec of the output
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the output is finalised.
pub fn convert_gene_file_to_v5(
    gene_in: &str,
    gene_out: &str,
    cell_lib: &[u64],
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let start = Instant::now();
    let reader = ParallelSparseReader::open_any(gene_in)?;
    require_pre_v5(&reader)?;
    if !reader.is_gene_based() {
        return Err(ScDataErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    let dims = reader.dims();
    let from_v3 = reader.file_version() == SC_FILE_VERSION_V3;
    let target_size = reader.target_size().unwrap_or(0.0);
    let order = reader.chunk_order()?;
    let lib_f32: Vec<f32> = cell_lib.iter().map(|&l| l as f32).collect();

    let writer = CellGeneSparseWriter::new(
        gene_out,
        false,
        dims.total_cells,
        dims.total_genes,
        target_size,
    )?;
    let mut writer = writer.with_codec(codec);
    let format = writer.format();

    let mut stored_norm = 0usize;
    for batch in order.chunks(CONVERT_GENE_BATCH) {
        let genes = reader.read_gene_parallel(batch)?;
        let payloads = genes
            .into_par_iter()
            .map(|mut gene| {
                if from_v3 {
                    let raw: Vec<u32> = gene.data_raw.iter().collect();
                    // a cell outside the table yields NaN, which never matches,
                    // and `to_v4_bytes` stores the layer
                    snap_v3_norm(&mut gene.data_norm, |k| {
                        let lib = lib_f32.get(gene.indices[k] as usize).copied();
                        lib.map_or(f32::NAN, |l| norm_value(raw[k] as f32, l, target_size))
                    });
                }
                Ok((gene.original_index, gene.to_v4_bytes(&lib_f32, &format)?))
            })
            .collect::<Result<Vec<_>, ScDataErrors>>()?;
        for (_, bytes) in &payloads {
            stored_norm += V4Header::parse(bytes)?.norm_stored() as usize;
        }
        writer.write_compressed_gene_chunks(&payloads)?;
    }
    writer.set_cell_library_table(cell_lib.to_vec())?;
    writer.finalise()?;

    if verbose {
        println!(
            "  Converted {} genes to v5 ({} with stored norm) in {:.2?}.",
            order.len().separate_with_underscores(),
            stored_norm.separate_with_underscores(),
            start.elapsed()
        );
    }

    Ok(())
}

/// Convert a v3 or v4 store to v5: the cell file, and the gene file when
/// given.
///
/// Inputs are never modified. Neither output may equal an input path.
///
/// ### Params
///
/// * `cell_in` - Path to the cell-based file
/// * `gene_in` - Path to the gene-based file, if any
/// * `cell_out` - Path of the v5 cell file
/// * `gene_out` - Path of the v5 gene file, required with `gene_in`
/// * `codec` - Body codec of both outputs
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once both outputs are finalised.
pub fn convert_store_to_v5(
    cell_in: &str,
    gene_in: Option<&str>,
    cell_out: &str,
    gene_out: Option<&str>,
    codec: ChunkCodec,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let lib = convert_cell_file_to_v5(cell_in, cell_out, codec, verbose)?;
    if let (Some(gene_in), Some(gene_out)) = (gene_in, gene_out) {
        convert_gene_file_to_v5(gene_in, gene_out, &lib, codec, verbose)?;
    }
    Ok(())
}

/// Replace a v3 or v4 store by its v5 conversion, in place.
///
/// Converts into `<path>.v5.tmp` next to each file first and only touches the
/// originals once every conversion has succeeded, so a failure leaves the
/// store as it was. Each original is then renamed to `<path>.v<n>.bak` (`n`
/// its version) when `keep_backup` is set, and deleted otherwise.
///
/// ### Params
///
/// * `cell_path` - Path to the cell-based file
/// * `gene_path` - Path to the gene-based file, if any
/// * `codec` - Body codec of the converted files
/// * `keep_backup` - Keep the originals as `<path>.v<n>.bak`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the converted files are in place.
pub fn upgrade_store_to_v5(
    cell_path: &str,
    gene_path: Option<&str>,
    codec: ChunkCodec,
    keep_backup: bool,
    verbose: bool,
) -> Result<(), ScDataErrors> {
    let tmp = |p: &str| format!("{p}.v5.tmp");
    let cleanup = |paths: &[Option<&str>]| {
        for p in paths.iter().flatten() {
            let _ = std::fs::remove_file(tmp(p));
        }
    };

    let mut versions = Vec::new();
    let result = (|| {
        let cell_tmp = tmp(cell_path);
        let lib = convert_cell_file_to_v5(cell_path, &cell_tmp, codec, verbose)?;
        if let Some(gene_path) = gene_path {
            convert_gene_file_to_v5(gene_path, &tmp(gene_path), &lib, codec, verbose)?;
        }
        // the converted files must open and agree with the originals
        for p in std::iter::once(cell_path).chain(gene_path) {
            let old = ParallelSparseReader::open_any(p)?;
            let new = ParallelSparseReader::new(&tmp(p))?;
            if (old.dims(), old.n_chunks()) != (new.dims(), new.n_chunks()) {
                return Err(ScDataErrors::V4ChunkCorrupt(format!(
                    "converted {p} does not match the original"
                )));
            }
            versions.push(old.file_version());
        }
        Ok(())
    })();
    if let Err(e) = result {
        cleanup(&[Some(cell_path), gene_path]);
        return Err(e);
    }

    for (p, version) in std::iter::once(cell_path).chain(gene_path).zip(versions) {
        if keep_backup {
            std::fs::rename(p, format!("{p}.v{version}.bak"))?;
        }
        std::fs::rename(tmp(p), p)?;
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::reader::peek_file_version;
    use crate::store::v4::tests::{
        N_CELLS, N_GENES, Scratch, TARGET, assert_cells_equal, assert_genes_equal, norm_bits,
        write_legacy_store,
    };
    use indexmap::IndexSet;

    fn assert_sparse_equal(
        a: &CompressedSparseData2<f32, f32>,
        b: &CompressedSparseData2<f32, f32>,
        exact: bool,
    ) {
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
        assert_eq!(
            a.data_2.as_deref().map(|n| norm_bits(n, exact)),
            b.data_2.as_deref().map(|n| norm_bits(n, exact))
        );
    }

    /// Every read path of a converted store matches the v3 or v4 original, for
    /// each codec and with a zero target size (norm always stored).
    fn check_conversion(version: u32, codec: ChunkCodec, target: f32) {
        let exact = version == SC_FILE_VERSION_V4;
        let tag = format!("v{version}_{codec:?}_{}", target as u32);
        let s = Scratch::new(&tag, 4);
        write_legacy_store(version, s.path(0), s.path(1), target);
        convert_store_to_v5(
            s.path(0),
            Some(s.path(1)),
            s.path(2),
            Some(s.path(3)),
            codec,
            false,
        )
        .unwrap();

        let (cells_old, cells_new) = (
            ParallelSparseReader::open_any(s.path(0)).unwrap(),
            ParallelSparseReader::new(s.path(2)).unwrap(),
        );
        let (genes_old, genes_new) = (
            ParallelSparseReader::open_any(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(3)).unwrap(),
        );
        assert_eq!(cells_old.file_version(), version);
        assert_eq!(cells_new.file_version(), SC_FILE_VERSION);
        assert_eq!(genes_new.file_version(), SC_FILE_VERSION);
        assert_eq!(cells_new.target_size(), cells_old.target_size());

        let all_cells: Vec<usize> = (0..N_CELLS).collect();
        let all_genes: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in cells_old
            .read_cells_parallel(&all_cells)
            .unwrap()
            .iter()
            .zip(&cells_new.read_cells_parallel(&all_cells).unwrap())
        {
            assert_cells_equal(a, b, exact);
        }
        for (a, b) in genes_old
            .read_gene_parallel(&all_genes)
            .unwrap()
            .iter()
            .zip(&genes_new.read_gene_parallel(&all_genes).unwrap())
        {
            assert_genes_equal(a, b, exact);
        }

        let shuffled: Vec<usize> = (0..N_CELLS).rev().step_by(3).collect();
        let keep: IndexSet<u32> = (0..N_CELLS as u32).filter(|c| c % 4 != 1).collect();
        for layer in [
            DataLayerReturn::Raw,
            DataLayerReturn::Norm,
            DataLayerReturn::BothLayers,
        ] {
            assert_sparse_equal(
                &cells_old
                    .read_cells_sparse::<f32>(&shuffled, &layer)
                    .unwrap(),
                &cells_new
                    .read_cells_sparse::<f32>(&shuffled, &layer)
                    .unwrap(),
                exact,
            );
            for subset in [None, Some(&keep)] {
                assert_sparse_equal(
                    &genes_old
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                    &genes_new
                        .read_genes_sparse::<f32>(&all_genes, subset, &layer)
                        .unwrap(),
                    exact,
                );
            }
        }

        for (a, b) in genes_old
            .read_gene_parallel_filtered(&all_genes, &keep)
            .unwrap()
            .iter()
            .zip(
                &genes_new
                    .read_gene_parallel_filtered(&all_genes, &keep)
                    .unwrap(),
            )
        {
            assert_genes_equal(a, b, exact);
        }
        assert_eq!(
            cells_old.read_cell_library_sizes(&all_cells).unwrap(),
            cells_new.read_cell_library_sizes(&all_cells).unwrap()
        );
        assert_eq!(
            genes_old.read_gene_nnz(&all_genes).unwrap(),
            genes_new.read_gene_nnz(&all_genes).unwrap()
        );
        assert_eq!(cells_old.n_chunks(), cells_new.n_chunks());
    }

    #[test]
    fn test_v3_to_v5_round_trip_none() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::None, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_lz4() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Lz4, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_zstd() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Zstd, TARGET);
    }

    #[test]
    fn test_v3_to_v5_round_trip_unknown_target_stores_norm() {
        check_conversion(SC_FILE_VERSION_V3, ChunkCodec::Lz4, 0.0);
    }

    #[test]
    fn test_v4_to_v5_round_trip_is_exact() {
        check_conversion(SC_FILE_VERSION_V4, ChunkCodec::Lz4, TARGET);
        check_conversion(SC_FILE_VERSION_V4, ChunkCodec::Zstd, 0.0);
    }

    #[test]
    fn test_v5_drops_norm_and_shrinks() {
        let s = Scratch::new("shrink", 4);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_store_to_v5(
            s.path(0),
            Some(s.path(1)),
            s.path(2),
            Some(s.path(3)),
            ChunkCodec::Lz4,
            false,
        )
        .unwrap();
        let size = |i: usize| std::fs::metadata(s.path(i)).unwrap().len();
        assert!(size(2) < size(0), "cell file {} vs {}", size(2), size(0));
        assert!(size(3) < size(1), "gene file {} vs {}", size(3), size(1));
    }

    #[test]
    fn test_default_writer_emits_v5_and_converter_refuses_it() {
        let s = Scratch::new("guards", 2);
        let mut writer = CellGeneSparseWriter::new(s.path(0), true, 1, 3, TARGET).unwrap();
        writer
            .write_cell_chunk(CsrCellChunk::from_data(&[1u32], &[0u32], 0, TARGET, true))
            .unwrap();
        writer.finalise().unwrap();
        assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
        assert!(matches!(
            convert_cell_file_to_v5(s.path(0), s.path(1), ChunkCodec::Lz4, false),
            Err(ScDataErrors::FileVersionMismatch { .. })
        ));
    }

    #[test]
    fn test_default_gene_file_is_v5_and_matches_the_v3_one() {
        let s = Scratch::new("genefile", 4);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_cell_file_to_v5(s.path(0), s.path(2), ChunkCodec::Lz4, false).unwrap();
        crate::store::gene_file::write_gene_file(s.path(2), s.path(3), Some(500), false).unwrap();
        assert_eq!(peek_file_version(s.path(3)).unwrap(), SC_FILE_VERSION);

        let (v3, v5) = (
            ParallelSparseReader::open_any(s.path(1)).unwrap(),
            ParallelSparseReader::new(s.path(3)).unwrap(),
        );
        let all: Vec<usize> = (0..N_GENES).collect();
        for (a, b) in v3
            .read_gene_parallel(&all)
            .unwrap()
            .iter()
            .zip(&v5.read_gene_parallel(&all).unwrap())
        {
            assert_genes_equal(a, b, false);
        }
    }

    #[test]
    fn test_upgrade_store_replaces_files_and_keeps_backups() {
        for version in [SC_FILE_VERSION_V3, SC_FILE_VERSION_V4] {
            let s = Scratch::new(&format!("upgrade{version}"), 2);
            write_legacy_store(version, s.path(0), s.path(1), TARGET);
            let cells_before = ParallelSparseReader::open_any(s.path(0))
                .unwrap()
                .get_all_cells()
                .unwrap();
            upgrade_store_to_v5(s.path(0), Some(s.path(1)), ChunkCodec::Lz4, true, false).unwrap();

            assert_eq!(peek_file_version(s.path(0)).unwrap(), SC_FILE_VERSION);
            assert_eq!(peek_file_version(s.path(1)).unwrap(), SC_FILE_VERSION);
            let backup = format!("{}.v{version}.bak", s.path(0));
            assert_eq!(peek_file_version(&backup).unwrap(), version);
            for (a, b) in cells_before.iter().zip(
                &ParallelSparseReader::new(s.path(0))
                    .unwrap()
                    .get_all_cells()
                    .unwrap(),
            ) {
                assert_cells_equal(a, b, version == SC_FILE_VERSION_V4);
            }
            for p in [backup, format!("{}.v{version}.bak", s.path(1))] {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    #[test]
    fn test_archive_round_trips_a_v5_cell_file() {
        let s = Scratch::new("archive", 6);
        write_legacy_store(SC_FILE_VERSION_V3, s.path(0), s.path(1), TARGET);
        convert_cell_file_to_v5(s.path(0), s.path(2), ChunkCodec::Lz4, false).unwrap();
        crate::store::archive::archive_cell_file(s.path(2), s.path(3), 3, false).unwrap();
        crate::store::archive::restore_archive(s.path(3), s.path(4), s.path(5), Some(500), false)
            .unwrap();

        let original = ParallelSparseReader::new(s.path(2)).unwrap();
        let restored = ParallelSparseReader::new(s.path(4)).unwrap();
        let all: Vec<usize> = (0..N_CELLS).collect();
        for (a, b) in original
            .read_cells_parallel(&all)
            .unwrap()
            .iter()
            .zip(&restored.read_cells_parallel(&all).unwrap())
        {
            assert_cells_equal(a, b, false);
        }
    }
}
