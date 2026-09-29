//! Builds the gene-based (CSC) binary from the cell-based (CSR) one.
//!
//! A counting-sort transpose streamed off the cell file. One pass counts the
//! non-zeros per `(cell chunk, gene)`; genes are then cut into contiguous
//! phases that fit a non-zero budget, and each phase re-reads the cells and
//! scatters its genes straight into flat buffers. Cells are visited in
//! ascending order within a chunk and chunks are ordered, so every gene's run
//! comes out sorted by cell without a sort.

use rayon::prelude::*;
use std::time::Instant;
use thousands::Separable;

use crate::core::math::sparse::ScatterPtr;
use crate::prelude::*;
use crate::single_cell::sc_data::data_io::*;

/// Cells decoded per read inside one worker's chunk.
///
/// Every worker holds one batch at a time, so peak decoded memory is roughly
/// `n_threads * GENE_FILE_READ_BATCH * mean_nnz_per_cell * 10 B`. At 16
/// threads and 1,200 nnz per cell that is about 400 MB, which stays well
/// under the phase buffers it feeds. [`CELL_BATCH_SIZE`] is sized for a
/// single reader and would be 25x that here.
const GENE_FILE_READ_BATCH: usize = 2_048;

/// Genes serialised and compressed per parallel batch before being appended.
///
/// Bounds the compressed payloads held at once to a slice of the phase rather
/// than the whole phase.
const GENE_FILE_WRITE_BATCH: usize = 1_024;

/// Stored bytes per non-zero in the phase buffers: raw `u32`, norm `F16`, cell
/// index `u32`.
pub const GENE_FILE_BYTES_PER_NNZ: usize = 10;

/////////////
// Helpers //
/////////////

/// Count the non-zeros per `(chunk, gene)`.
///
/// ### Params
///
/// * `reader` - Reader over the cell-based file
/// * `chunks` - Half-open cell ranges, one per worker
/// * `n_genes` - Number of genes in the file
///
/// ### Returns
///
/// Chunk-major histogram, `chunks.len() * n_genes` long.
fn count_gene_nnz(
    reader: &ParallelSparseReader,
    chunks: &[(usize, usize)],
    n_genes: usize,
) -> Result<Vec<usize>, BixverseErrors> {
    let mut counts = vec![0usize; chunks.len() * n_genes];

    chunks
        .par_iter()
        .zip(counts.par_chunks_mut(n_genes.max(1)))
        .try_for_each(|(&(start, end), counts)| {
            for batch_start in (start..end).step_by(GENE_FILE_READ_BATCH) {
                let batch_end = (batch_start + GENE_FILE_READ_BATCH).min(end);
                for cell in reader.read_cells_range(batch_start, batch_end)? {
                    for &gene in &cell.indices {
                        counts[gene as usize] += 1;
                    }
                }
            }
            Ok::<(), BixverseErrors>(())
        })?;

    Ok(counts)
}

/// Cut the genes into contiguous phases that each fit the budget.
///
/// A gene larger than the budget on its own gets a phase to itself.
///
/// ### Params
///
/// * `gene_nnz` - Non-zeros per gene
/// * `max_nnz` - Budget per phase, `None` for a single phase
///
/// ### Returns
///
/// Half-open gene ranges covering `0..gene_nnz.len()`.
fn plan_phases(gene_nnz: &[usize], max_nnz: Option<usize>) -> Vec<(usize, usize)> {
    let n_genes = gene_nnz.len();
    let Some(max_nnz) = max_nnz else {
        return vec![(0, n_genes)];
    };

    let mut phases = Vec::new();
    let mut start = 0;
    let mut acc = 0usize;
    for (gene, &nnz) in gene_nnz.iter().enumerate() {
        if gene > start && acc + nnz > max_nnz {
            phases.push((start, gene));
            start = gene;
            acc = 0;
        }
        acc += nnz;
    }
    if start < n_genes || phases.is_empty() {
        phases.push((start, n_genes));
    }
    phases
}

/////////////////
// Gene writer //
/////////////////

/// Write the gene-based file for a cell-based one.
///
/// Replaces the in-memory, hash-map and per-phase re-read conversions that
/// used to live in the R wrapper. The output is identical to transposing the
/// full matrix: per gene, cells ascend and the normalised layer is carried
/// over bit for bit from the cell file.
///
/// Memory is about `max_nnz_in_memory * GENE_FILE_BYTES_PER_NNZ` for the phase
/// buffers, plus the decoded read batches (see [`GENE_FILE_READ_BATCH`]) and
/// one write batch of compressed payloads. The cell file is decoded once for
/// the count and once per phase.
///
/// ### Params
///
/// * `cell_path` - Path to the cell-based binary
/// * `gene_path` - Path of the gene-based binary to write
/// * `max_nnz_in_memory` - Non-zeros held per phase. `None` converts in a
///   single phase.
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `Ok(())` once the gene file is finalised.
pub fn write_gene_file(
    cell_path: &str,
    gene_path: &str,
    max_nnz_in_memory: Option<usize>,
    verbose: bool,
) -> Result<(), BixverseErrors> {
    let reader = ParallelSparseReader::new(cell_path)?;
    let header = reader.get_header();
    let n_cells = header.total_cells;
    let n_genes = header.total_genes;
    let chunks = thread_chunks(n_cells);

    let start_total = Instant::now();

    let counts = count_gene_nnz(&reader, &chunks, n_genes)?;
    let mut gene_nnz = vec![0usize; n_genes];
    for chunk_counts in counts.chunks(n_genes.max(1)) {
        for (total, &c) in gene_nnz.iter_mut().zip(chunk_counts) {
            *total += c;
        }
    }

    let phases = plan_phases(&gene_nnz, max_nnz_in_memory);
    let max_phase_nnz = phases
        .iter()
        .map(|&(g0, g1)| gene_nnz[g0..g1].iter().sum::<usize>())
        .max()
        .unwrap_or(0);

    if verbose {
        println!(
            "  Counted {} non-zeros over {} genes in {:.2?}; converting in {} phase(s).",
            gene_nnz.iter().sum::<usize>().separate_with_underscores(),
            n_genes.separate_with_underscores(),
            start_total.elapsed(),
            phases.len()
        );
    }

    // The gene file inherits the cell file's normalisation, so it has to carry
    // the same target size in its header.
    let mut writer = CellGeneSparseWriter::new(
        gene_path,
        false,
        n_cells,
        n_genes,
        reader.target_size().unwrap_or(0.0),
    )?;

    let mut raw = vec![0u32; max_phase_nnz];
    let mut norm = vec![F16::default(); max_phase_nnz];
    let mut cell_idx = vec![0u32; max_phase_nnz];

    for (phase_i, &(g0, g1)) in phases.iter().enumerate() {
        let start_phase = Instant::now();
        let n_phase = g1 - g0;

        // gene_start[g - g0] is where gene g's run begins in the buffers
        let mut gene_start = vec![0usize; n_phase + 1];
        for g in g0..g1 {
            gene_start[g - g0 + 1] = gene_start[g - g0] + gene_nnz[g];
        }

        // per-chunk write cursors: chunk c writes gene g after chunks < c
        let mut cursors = vec![0usize; chunks.len() * n_phase];
        let mut running = gene_start[..n_phase].to_vec();
        for (c, cursor_row) in cursors.chunks_mut(n_phase.max(1)).enumerate() {
            let chunk_counts = &counts[c * n_genes + g0..c * n_genes + g1];
            for ((cur, run), &cnt) in cursor_row
                .iter_mut()
                .zip(running.iter_mut())
                .zip(chunk_counts)
            {
                *cur = *run;
                *run += cnt;
            }
        }

        let raw_ptr = ScatterPtr(raw.as_mut_ptr());
        let norm_ptr = ScatterPtr(norm.as_mut_ptr());
        let idx_ptr = ScatterPtr(cell_idx.as_mut_ptr());

        chunks
            .par_iter()
            .zip(cursors.par_chunks_mut(n_phase.max(1)))
            .try_for_each(|(&(start, end), cursor)| {
                for batch_start in (start..end).step_by(GENE_FILE_READ_BATCH) {
                    let batch_end = (batch_start + GENE_FILE_READ_BATCH).min(end);
                    let cells = reader.read_cells_range(batch_start, batch_end)?;
                    for (offset, cell) in cells.iter().enumerate() {
                        let cell_id = (batch_start + offset) as u32;
                        // gene indices are sorted within a cell
                        let lo = cell.indices.partition_point(|&g| (g as usize) < g0);
                        let hi = cell.indices.partition_point(|&g| (g as usize) < g1);
                        for k in lo..hi {
                            let local = cell.indices[k] as usize - g0;
                            let pos = cursor[local];
                            cursor[local] += 1;
                            // SAFETY: `pos` lies in the run the counting pass
                            // reserved for (this chunk, this gene), which no
                            // other worker writes, and every run sits inside
                            // `0..max_phase_nnz`.
                            unsafe {
                                raw_ptr.write(pos, cell.data_raw.get(k));
                                norm_ptr.write(pos, cell.data_norm[k]);
                                idx_ptr.write(pos, cell_id);
                            }
                        }
                    }
                }
                Ok::<(), BixverseErrors>(())
            })?;

        let scatter_time = start_phase.elapsed();
        let start_write = Instant::now();

        for batch_start in (g0..g1).step_by(GENE_FILE_WRITE_BATCH) {
            let batch_end = (batch_start + GENE_FILE_WRITE_BATCH).min(g1);
            let payloads = (batch_start..batch_end)
                .into_par_iter()
                .map(|g| {
                    let s = gene_start[g - g0];
                    let e = gene_start[g - g0 + 1];
                    let data_norm = norm[s..e].to_vec();
                    // accumulate in f32, see `CscGeneChunk::from_conversion`
                    let avg_exp = F16::from_f32(data_norm.iter().map(|v| v.to_f32()).sum::<f32>());
                    let chunk = CscGeneChunk {
                        data_raw: RawCounts::from_u32_auto(&raw[s..e]),
                        data_norm,
                        avg_exp,
                        nnz: e - s,
                        indices: cell_idx[s..e].to_vec(),
                        original_index: g,
                        to_keep: true,
                    };
                    Ok((g, chunk.to_compressed_bytes()?))
                })
                .collect::<Result<Vec<_>, BixverseErrors>>()?;
            writer.write_compressed_gene_chunks(&payloads)?;
        }

        if verbose {
            println!(
                "  Phase {}/{}: genes {}-{}, scatter {:.2?}, compress + write {:.2?}",
                phase_i + 1,
                phases.len(),
                g0,
                g1,
                scatter_time,
                start_write.elapsed()
            );
        }
    }

    writer.finalise()?;

    if verbose {
        println!(
            "  Conversion into gene-friendly format done: {:.2?}",
            start_total.elapsed()
        );
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

    /// Drop implementation for [`TempBin`]. Errors are ignored: the file may
    /// already be gone, and this runs during unwind.
    impl Drop for TempBin {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl TempBin {
        /// Reserve a uniquely named scratch file in the system temp directory.
        ///
        /// ### Params
        ///
        /// * `name` - Test-unique suffix.
        ///
        /// ### Returns
        ///
        /// The guard.
        fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("bixverse_gene_file_io_{name}.bin")))
        }

        /// Path of the guarded file as a `&str`.
        fn path(&self) -> &str {
            self.0.to_str().expect("temp path is valid UTF-8")
        }
    }

    /// Write a deterministic cell file. Cell `i` expresses every gene `g` with
    /// `(i * 7 + g * 3) % 5 == 0`, count `1 + (i + g) % 4`; gene 3 is left
    /// empty and cell 2 has no genes at all.
    ///
    /// ### Params
    ///
    /// * `path` - Where to write
    /// * `n_cells` - Number of cells
    /// * `n_genes` - Number of genes
    fn write_cells(path: &str, n_cells: usize, n_genes: usize) {
        let mut writer = CellGeneSparseWriter::new(path, true, n_cells, n_genes, 1e4).unwrap();
        for i in 0..n_cells {
            let (mut data, mut idx) = (Vec::new(), Vec::new());
            if i != 2 {
                for g in 0..n_genes {
                    if g != 3 && (i * 7 + g * 3) % 5 == 0 {
                        data.push(1 + ((i + g) % 4) as u32);
                        idx.push(g as u32);
                    }
                }
            }
            writer
                .write_cell_chunk(CsrCellChunk::from_data(&data, &idx, i, 1e4, true))
                .unwrap();
        }
        writer.finalise().unwrap();
    }

    /// Reference gene chunks from the old in-memory algorithm: flatten every
    /// cell and transpose with [`transpose_sparse`].
    ///
    /// ### Params
    ///
    /// * `cell_path` - Cell-based file
    ///
    /// ### Returns
    ///
    /// `(raw, norm bits, cell indices)` per gene.
    #[allow(clippy::type_complexity)]
    fn reference(cell_path: &str) -> Vec<(Vec<u32>, Vec<u16>, Vec<u32>)> {
        let reader = ParallelSparseReader::new(cell_path).unwrap();
        let h = reader.get_header();
        let (mut data, mut data2, mut idx, mut indptr) = (vec![], vec![], vec![], vec![0u32]);
        for cell in reader.get_all_cells().unwrap() {
            data.extend(cell.data_raw.iter());
            data2.extend(cell.data_norm.iter().map(|v| v.to_f32()));
            idx.extend(cell.indices.iter().copied());
            indptr.push(data.len() as u32);
        }
        let csr = CompressedSparseData2::new_csr(
            &data,
            &idx,
            &indptr,
            Some(&data2),
            (h.total_cells, h.total_genes),
        );
        let csc = csr.transform();
        let d2 = csc.data_2.as_ref().unwrap();
        (0..h.total_genes)
            .map(|g| {
                let (s, e) = (csc.indptr[g] as usize, csc.indptr[g + 1] as usize);
                (
                    csc.data[s..e].to_vec(),
                    d2[s..e]
                        .iter()
                        .map(|&v| F16::from_f32(v).to_bits())
                        .collect(),
                    csc.indices[s..e].to_vec(),
                )
            })
            .collect()
    }

    /// Read every gene of a gene file back.
    ///
    /// ### Params
    ///
    /// * `gene_path` - Gene-based file
    ///
    /// ### Returns
    ///
    /// `(raw, norm bits, cell indices)` per gene.
    #[allow(clippy::type_complexity)]
    fn read_genes(gene_path: &str) -> Vec<(Vec<u32>, Vec<u16>, Vec<u32>)> {
        let reader = ParallelSparseReader::new(gene_path).unwrap();
        reader
            .get_all_genes()
            .unwrap()
            .into_iter()
            .map(|g| {
                (
                    g.data_raw.iter().collect(),
                    g.data_norm.iter().map(|v| v.to_bits()).collect(),
                    g.indices,
                )
            })
            .collect()
    }

    #[test]
    fn test_write_gene_file_matches_transpose() {
        let cells = TempBin::new("match_cells");
        let genes = TempBin::new("match_genes");
        write_cells(cells.path(), 57, 11);

        write_gene_file(cells.path(), genes.path(), None, false).unwrap();

        assert_eq!(read_genes(genes.path()), reference(cells.path()));
    }

    #[test]
    fn test_write_gene_file_multi_phase_identical() {
        let cells = TempBin::new("phase_cells");
        let single = TempBin::new("phase_single");
        let multi = TempBin::new("phase_multi");
        write_cells(cells.path(), 57, 11);

        write_gene_file(cells.path(), single.path(), None, false).unwrap();
        // budget below the largest gene forces one phase per gene
        write_gene_file(cells.path(), multi.path(), Some(1), false).unwrap();

        assert_eq!(read_genes(single.path()), read_genes(multi.path()));
    }

    #[test]
    fn test_write_gene_file_empty_gene_and_header() {
        let cells = TempBin::new("empty_cells");
        let genes = TempBin::new("empty_genes");
        write_cells(cells.path(), 20, 6);

        write_gene_file(cells.path(), genes.path(), Some(15), false).unwrap();

        let reader = ParallelSparseReader::new(genes.path()).unwrap();
        assert!(!reader.is_cell_based());
        assert_eq!(reader.get_header().total_cells, 20);
        assert_eq!(reader.get_header().total_genes, 6);
        assert_eq!(reader.target_size(), Some(1e4));
        let g3 = &read_genes(genes.path())[3];
        assert!(g3.0.is_empty() && g3.2.is_empty());
    }

    #[test]
    fn test_plan_phases() {
        assert_eq!(plan_phases(&[3, 3, 3], None), vec![(0, 3)]);
        assert_eq!(plan_phases(&[3, 3, 3], Some(6)), vec![(0, 2), (2, 3)]);
        assert_eq!(plan_phases(&[10, 1, 1], Some(5)), vec![(0, 1), (1, 3)]);
        assert_eq!(plan_phases(&[], Some(5)), vec![(0, 0)]);
    }
}
