use bincode::{Decode, Encode, config, decode_from_slice, serde::encode_to_vec};
use half::f16;
use indexmap::IndexSet;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use memmap2::MmapOptions;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    marker::Sync,
    path::Path,
    sync::Arc,
};

use crate::prelude::*;

//////////////////
// Cell Quality //
//////////////////

/// Structure to store QC information on cells
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CellQuality {
    /// Indices of which cells to keep.
    pub cell_indices: Vec<usize>,
    /// Indices of which genes to keep
    pub gene_indices: Vec<usize>,
    /// Library size of cells
    pub lib_size: Vec<usize>,
    /// Number of cells expressing this gene
    pub nnz: Vec<usize>,
}

impl CellQuality {
    /// Update the internal cell indices
    ///
    /// ### Params
    ///
    /// * `cell_indices` - Vector of cell indices to keep
    pub fn set_cell_indices(&mut self, cell_indices: &[usize]) {
        self.cell_indices = cell_indices.to_vec();
    }

    /// Update the internal gene indices
    ///
    /// ### Params
    ///
    /// * `gene_indices` - Vector of gene indices to keep
    pub fn set_gene_indices(&mut self, gene_indices: &[usize]) {
        self.gene_indices = gene_indices.to_vec();
    }
}

/// Structure that stores minimum QC thresholds/info for single cell
#[derive(Clone, Debug)]
pub struct MinCellQuality {
    /// Minimum number of unique genes per cell (or spot for Visium type
    /// technology).
    pub min_unique_genes: usize,
    /// Minimum library size per cell (or spot for Visium type technology).
    pub min_lib_size: usize,
    /// Minimum library size per cell (or spot for Visium type technology)/
    pub min_cells: usize,
    /// Target size for the library size normalisation (typical values 1e4 -
    /// 1e5 single cell.
    pub target_size: f32,
}

///////////////////////////
// Sparse data streaming //
///////////////////////////

///////////////////////
// CellOnFileQuality //
///////////////////////

/// CellOnFileQuality
///
/// This structure is being generate after a first scan of the file on disk and
/// defining which cells and genes to actually read in.
#[derive(Debug, Clone)]
pub struct CellOnFileQuality {
    /// Vector of indices of the cells to keep.
    pub cells_to_keep: Vec<usize>,
    /// Vector of indices of the genes to keep.
    pub genes_to_keep: Vec<usize>,
    /// HashSet of the indices to keep (for look-ups).
    pub cells_to_keep_set: FxHashSet<usize>,
    /// HashSet of the genes to keep (for look-ups).
    pub genes_to_keep_set: FxHashSet<usize>,
    /// Mapping of the old indices to the new indices for the cells.
    pub cell_old_to_new: FxHashMap<usize, usize>,
    /// Mapping of old indices to new indices for the genes.
    pub gene_old_to_new: FxHashMap<usize, usize>,
}

impl CellOnFileQuality {
    /// Create a new CellOnFileQuality structure
    ///
    /// ### Params
    ///
    /// * cells_to_keep - Index positions of the cells to keep
    /// * genes_to_keep - Index positions of the genes to keep
    ///
    /// ### Returns
    ///
    /// Initiliased self
    pub fn new(cells_to_keep: Vec<usize>, genes_to_keep: Vec<usize>) -> Self {
        Self {
            cells_to_keep,
            genes_to_keep,
            cells_to_keep_set: FxHashSet::default(),
            genes_to_keep_set: FxHashSet::default(),
            cell_old_to_new: FxHashMap::default(),
            gene_old_to_new: FxHashMap::default(),
        }
    }

    /// Generate internally the sets and maps
    pub fn generate_maps_sets(&mut self) {
        let cells_to_keep_set: FxHashSet<usize> = self.cells_to_keep.iter().cloned().collect();
        let genes_to_keep_set: FxHashSet<usize> = self.genes_to_keep.iter().cloned().collect();
        let cell_old_to_new: FxHashMap<usize, usize> = self
            .cells_to_keep
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();
        let gene_old_to_new: FxHashMap<usize, usize> = self
            .genes_to_keep
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        self.cells_to_keep_set = cells_to_keep_set;
        self.genes_to_keep_set = genes_to_keep_set;
        self.cell_old_to_new = cell_old_to_new;
        self.gene_old_to_new = gene_old_to_new
    }
}

//////////////////
// CsrCellChunk //
//////////////////

/// CsrCellChunk
///
/// This structure is designed to store the data of a single cell in a
/// CSR-like format optimised for rapid access on disk.
#[derive(Debug)]
pub struct CsrCellChunk {
    /// Vector of the raw counts of this cell. This is limited to u16, limiting
    /// the max raw counts per gene to 65_535
    pub data_raw: Vec<u16>,
    /// Vector of the norm counts of this cell. A lossy compression for f16 is
    /// applied.
    pub data_norm: Vec<F16>,
    /// Total library size/UMI counts of the cell.
    pub library_size: usize,
    /// Index positions of the genes
    pub indices: Vec<u16>,
    /// Original index in the data
    pub original_index: usize,
    /// Flag if the cell should be kept. (Not used at the moment.)
    pub to_keep: bool,
}

impl CsrCellChunk {
    /// Function to generate the chunk from R data
    ///
    /// Assumes columns = genes, rows = cells.
    ///
    /// ### Params
    ///
    /// * `data` - The raw counts present in this cell
    /// * `col_idx` - The column indices where the gene is expressed.
    /// * `original_index` - Original row index in the matrix
    /// * `size_factor` - To which size to normalise to. 1e6 -> CPM normalisation.
    ///
    /// ### Returns
    ///
    /// The `CsrCellChunk` for this cell.
    pub fn from_data<T, U>(
        data: &[T],
        col_idx: &[U],
        original_index: usize,
        size_factor: f32,
        to_keep: bool,
    ) -> Self
    where
        T: ToF32AndU16,
        U: ToF32AndU16,
    {
        let data_f32 = data.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>();
        let sum = data_f32.iter().sum::<f32>();
        let data_norm: Vec<F16> = data_f32
            .into_iter()
            .map(|x| {
                let norm = (x / sum * size_factor).ln_1p();
                F16::from(f16::from_f32(norm))
            })
            .collect();
        Self {
            data_raw: data.iter().map(|&x| x.to_u16()).collect::<Vec<u16>>(),
            data_norm,
            library_size: sum as usize,
            indices: col_idx.iter().map(|&x| x.to_u16()).collect::<Vec<u16>>(),
            original_index,
            to_keep,
        }
    }

    /// Write directly to bytes on disk
    ///
    /// ### Params
    ///
    /// * `writer` - Something that can write, i.e., has the implementation
    ///   `Write`
    pub fn write_to_bytes(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&(self.data_raw.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data_norm.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.indices.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.library_size as u64).to_le_bytes())?;
        writer.write_all(&(self.original_index as u64).to_le_bytes())?;
        // include some padding
        writer.write_all(&[self.to_keep as u8, 0, 0, 0])?;

        // unsafe fun with direct write to disk
        let data_raw_bytes = unsafe {
            std::slice::from_raw_parts(self.data_raw.as_ptr() as *const u8, self.data_raw.len() * 2)
        };
        writer.write_all(data_raw_bytes)?;

        let data_norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.data_norm.as_ptr() as *const u8,
                self.data_norm.len() * 2,
            )
        };
        writer.write_all(data_norm_bytes)?;

        let col_indices_bytes = unsafe {
            std::slice::from_raw_parts(self.indices.as_ptr() as *const u8, self.indices.len() * 2)
        };
        writer.write_all(col_indices_bytes)?;

        Ok(())
    }

    /// Read data from buffer
    ///
    /// ### Params
    ///
    /// * `buffer` - A slice of u8's representing the buffer
    ///
    /// ### Return
    ///
    /// The `CsrCellChunk`
    pub fn read_from_buffer(buffer: &[u8]) -> std::io::Result<Self> {
        if buffer.len() < 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Buffer too small for header",
            ));
        }

        // parse header
        let header = &buffer[0..32];

        let data_raw_len =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let data_norm_len =
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
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
        let data_end = data_start + data_raw_len * 2;
        let norm_end = data_end + data_norm_len * 2;

        // direct transmutation for raw counts - no intermediate allocation
        let data_raw = unsafe {
            let ptr = buffer.as_ptr().add(data_start) as *const u16;
            std::slice::from_raw_parts(ptr, data_raw_len).to_vec()
        };

        // For F16, I need to convert
        let data_norm: Vec<F16> = unsafe {
            let ptr = buffer.as_ptr().add(data_end) as *const u16;
            std::slice::from_raw_parts(ptr, data_norm_len)
                .iter()
                .map(|&bits| F16::from_bits(bits))
                .collect()
        };

        let indices = unsafe {
            let ptr = buffer.as_ptr().add(norm_end) as *const u16;
            std::slice::from_raw_parts(ptr, col_indices_len).to_vec()
        };

        Ok(Self {
            data_raw,
            data_norm,
            library_size,
            indices,
            original_index,
            to_keep,
        })
    }

    /// Generate a vector of Chunks from CompressedSparseData2
    ///
    /// ### Params
    ///
    /// * `sparse_data` - The `CompressedSparseData2` (in CSR format!)
    /// * `min_genes` - Number of genes per cell to be included
    /// * `size_factor` - Size factor for normalisation. 1e6 -> CPM
    ///
    /// ### Returns
    ///
    /// A tuple of the `Vec<CsrCellChunk>` and if the cell should be kept.
    pub fn generate_chunks_sparse_data<T, U>(
        sparse_data: CompressedSparseData2<T, U>,
        cell_qc: MinCellQuality,
    ) -> (Vec<CsrCellChunk>, CellQuality)
    where
        T: Clone + Default + Into<u32> + Sync,
        U: Clone + Default,
    {
        let n_cells = sparse_data.indptr.len() - 1;
        let n_genes = sparse_data.shape.1;

        // count how many cells express each gene
        let mut no_cells_exp_gene = vec![0usize; n_genes];
        for i in 0..n_cells {
            let start_i = sparse_data.indptr[i];
            let end_i = sparse_data.indptr[i + 1];
            for &gene_idx in &sparse_data.indices[start_i..end_i] {
                no_cells_exp_gene[gene_idx] += 1;
            }
        }

        // filter genes
        let genes_to_keep: Vec<usize> = (0..n_genes)
            .filter(|&i| no_cells_exp_gene[i] >= cell_qc.min_cells)
            .collect();

        // create index mapping: old gene index -> new gene index
        // these f--king reindexing bugs -.-
        let mut gene_index_map = vec![None; n_genes];
        for (new_idx, &old_idx) in genes_to_keep.iter().enumerate() {
            gene_index_map[old_idx] = Some(new_idx);
        }

        // Process cells
        let results: Vec<_> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let start_i = sparse_data.indptr[i];
                let end_i = sparse_data.indptr[i + 1];

                let mut filtered_data = Vec::new();
                let mut filtered_indices = Vec::new();

                for idx in start_i..end_i {
                    let old_gene_idx = sparse_data.indices[idx];
                    if let Some(new_gene_idx) = gene_index_map[old_gene_idx] {
                        filtered_data.push(sparse_data.data[idx].clone().into());
                        filtered_indices.push(new_gene_idx);
                    }
                }

                let sum_data_i = filtered_data.iter().sum::<u32>() as usize;
                let nnz_i = filtered_indices.len();

                let to_keep_i =
                    (nnz_i >= cell_qc.min_unique_genes) && (sum_data_i >= cell_qc.min_lib_size);

                let chunk_i = CsrCellChunk::from_data(
                    &filtered_data,
                    &filtered_indices,
                    i,
                    cell_qc.target_size,
                    to_keep_i,
                );

                (chunk_i, sum_data_i, nnz_i, to_keep_i, i)
            })
            .collect();

        let mut res = Vec::with_capacity(n_cells);
        let mut lib_size = Vec::new();
        let mut nnz = Vec::new();
        let mut cells_to_keep = Vec::new();

        for (chunk, lib, n, keep, idx) in results {
            res.push(chunk);
            if keep {
                lib_size.push(lib);
                nnz.push(n);
                cells_to_keep.push(idx);
            }
        }

        let qc_data = CellQuality {
            cell_indices: cells_to_keep,
            gene_indices: genes_to_keep,
            lib_size,
            nnz,
        };

        (res, qc_data)
    }

    /// Helper function to get QC parameters for this cell
    ///
    /// ### Reutrns
    ///
    /// A tuple of `(no_genes, library_size)`
    pub fn get_qc_info(&self) -> (usize, usize) {
        (self.indices.len(), self.library_size)
    }

    /// Filter to keep only specified genes
    ///
    /// This will remove the genes that are not set to `true` in the boolean
    /// vector. This assumes that the indices of the genes and the booleans
    /// are the same!
    ///
    /// ### Params
    ///
    /// * `genes_to_keep` - Keep only the genes within this boolean vectors.
    pub fn filter_genes(&mut self, genes_to_keep: &[bool]) {
        let mut index_map = vec![None; genes_to_keep.len()];
        let mut new_idx = 0;
        for (old_idx, &keep) in genes_to_keep.iter().enumerate() {
            if keep {
                index_map[old_idx] = Some(new_idx);
                new_idx += 1;
            }
        }

        let mut new_indices = Vec::new();
        let mut new_raw = Vec::new();
        let mut new_norm = Vec::new();

        for (i, &gene_idx) in self.indices.iter().enumerate() {
            if let Some(new_gene_idx) = index_map[gene_idx as usize] {
                new_indices.push(new_gene_idx as u16);
                new_raw.push(self.data_raw[i]);
                new_norm.push(self.data_norm[i]);
            }
        }

        self.indices = new_indices;
        self.data_raw = new_raw;
        self.data_norm = new_norm;
    }
}

/// CscGeneChunk
///
/// This structure is designed to store the data of a single gene in a
/// CSC-like format optimised for rapid access on disk.
#[derive(Encode, Decode, Serialize, Deserialize, Debug)]
pub struct CscGeneChunk {
    /// Vector with the raw counts per cell/spot for this gene. The maximum
    /// counts per gene are limited to 65_535
    pub data_raw: Vec<u16>,
    /// Vector with normalised coutns per cell/spot for this gene. Lossy
    /// compression to f16 is applied.
    pub data_norm: Vec<F16>,
    /// Average expression of this gene in the data.
    pub avg_exp: F16,
    /// Number of cells expressing this gene
    pub nnz: usize,
    /// Indices of the cells expressing this gene
    pub indices: Vec<u32>,
    /// Original index from the data
    pub original_index: usize,
    /// Flag to indicate if gene shall be kept. Not in use at the moment.
    pub to_keep: bool,
}

impl CscGeneChunk {
    /// Helper function to generate the CscGeneChunk from converted data
    ///
    /// ### Params
    ///
    /// * `data_raw` - The raw counts for this gene
    /// * `data_norm` - The normalised counts for this gene.
    /// * `col_idx` - The column indices for which cells this gene is expressed.
    /// * `original_index` - Original row index
    /// * `to_keep` - Shall the gene be included in later analysis.
    ///
    /// ### Returns
    ///
    /// The `CscGeneChunk` for this gene.
    pub fn from_conversion(
        data_raw: &[u16],
        data_norm: &[F16],
        col_idx: &[usize],
        original_index: usize,
        to_keep: bool,
    ) -> Self {
        let avg_exp = data_norm.iter().sum::<F16>();
        let nnz = data_raw.len();

        Self {
            data_raw: data_raw.to_vec(),
            data_norm: data_norm.to_vec(),
            avg_exp,
            nnz,
            indices: col_idx.iter().map(|x| *x as u32).collect::<Vec<u32>>(),
            original_index,
            to_keep,
        }
    }

    /// Write directly to bytes on disk
    ///
    /// ### Params
    ///
    /// * `writer` - Something that can write, i.e., has the implementation
    ///   `Write`
    pub fn write_to_bytes(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&(self.data_raw.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data_norm.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.indices.len() as u32).to_le_bytes())?;
        writer.write_all(&self.avg_exp.to_le_bytes())?;
        // first padding
        writer.write_all(&[0, 0])?;
        writer.write_all(&(self.nnz as u64).to_le_bytes())?;
        writer.write_all(&(self.original_index as u64).to_le_bytes())?;
        // bit padding
        writer.write_all(&[self.to_keep as u8, 0, 0, 0])?;

        let data_raw_bytes = unsafe {
            std::slice::from_raw_parts(self.data_raw.as_ptr() as *const u8, self.data_raw.len() * 2)
        };
        writer.write_all(data_raw_bytes)?;

        let data_norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.data_norm.as_ptr() as *const u8,
                self.data_norm.len() * 2,
            )
        };
        writer.write_all(data_norm_bytes)?;

        let row_indices_bytes = unsafe {
            std::slice::from_raw_parts(self.indices.as_ptr() as *const u8, self.indices.len() * 4)
        };
        writer.write_all(row_indices_bytes)?;

        Ok(())
    }

    /// Read data from buffer
    ///
    /// ### Params
    ///
    /// * `buffer` - A slice of u8's representing the buffer
    ///
    /// ### Return
    ///
    /// The `CscGeneChunk`
    pub fn read_from_buffer(buffer: &[u8]) -> std::io::Result<Self> {
        if buffer.len() < 36 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Buffer too small for header",
            ));
        }

        // parse header
        let header = &buffer[0..36];
        let data_raw_len =
            u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let data_norm_len =
            u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
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
        let data_end = data_start + data_raw_len * 2;
        let norm_end = data_end + data_norm_len * 2;

        // direct transmutation for raw counts - no intermediate allocation
        let data_raw = unsafe {
            let ptr = buffer.as_ptr().add(data_start) as *const u16;
            std::slice::from_raw_parts(ptr, data_raw_len).to_vec()
        };

        // For F16, I need to convert
        let data_norm = unsafe {
            let ptr = buffer.as_ptr().add(data_end) as *const u16;
            let slice = std::slice::from_raw_parts(ptr, data_norm_len);
            let mut norm = Vec::with_capacity(data_norm_len);
            norm.extend(slice.iter().map(|&bits| F16::from_bits(bits)));
            norm
        };

        let indices = unsafe {
            let ptr = buffer.as_ptr().add(norm_end) as *const u32;
            std::slice::from_raw_parts(ptr, row_indices_len).to_vec()
        };

        Ok(Self {
            data_raw,
            data_norm,
            avg_exp,
            nnz,
            indices,
            original_index,
            to_keep,
        })
    }

    /// Helper function that allows to filter out cells
    ///
    /// ### Params
    ///
    /// * `cells_to_keep` - IndexSet with cell index positions to keep.
    pub fn filter_selected_cells(&mut self, cells_to_keep: &IndexSet<u32>) {
        // build reverse mapping: cell_id -> position in this gene's data
        let cell_positions: FxHashMap<u32, usize> = self
            .indices
            .iter()
            .enumerate()
            .map(|(pos, &cell_id)| (cell_id, pos))
            .collect();

        let mut new_data_raw = Vec::with_capacity(cells_to_keep.len());
        let mut new_data_norm = Vec::with_capacity(cells_to_keep.len());
        let mut new_row_indices = Vec::with_capacity(cells_to_keep.len());

        // tterate in cells_to_keep order (critical for PCA! tripped over this one...)
        for (new_row_idx, &cell_index) in cells_to_keep.iter().enumerate() {
            if let Some(&pos) = cell_positions.get(&cell_index) {
                new_data_raw.push(self.data_raw[pos]);
                new_data_norm.push(self.data_norm[pos]);
                new_row_indices.push(new_row_idx as u32);
            }
        }

        let nnz = new_data_raw.len();
        self.data_raw = new_data_raw;
        self.data_norm = new_data_norm;
        self.indices = new_row_indices;
        self.nnz = nnz;
    }

    /// Calculate the average gene expression given a set of cells
    ///
    /// ### Params
    ///
    /// * `cells_to_keep` - IndexSet with cell index positions to keep.
    ///
    /// ### Returns
    ///
    /// Tuple of `(gene_index, average expression)`.
    pub fn calculate_avg_exp(&self, cells_to_keep: &IndexSet<u32>) -> (usize, f32) {
        let sum: f32 = self
            .indices
            .iter()
            .zip(self.data_norm.iter())
            .filter(|(cell_idx, _)| cells_to_keep.contains(*cell_idx))
            .map(|(_, val)| val.to_f32())
            .sum();
        let avg = sum / cells_to_keep.len() as f32;

        (self.original_index, avg)
    }

    /// Transform the chunk to a sparse Axis of CSC type
    ///
    /// ### Params
    ///
    /// * `n_cells` - Number of cells represented in the data
    ///
    /// ### Returns
    ///
    /// `SparseAxis` with u16 in the main slot and f32 in the data_2 layer.
    pub fn to_sparse_axis(&self, n_cells: usize) -> SparseAxis<u16, f32> {
        SparseAxis::new_csc(
            self.indices.iter().map(|x| *x as usize).collect(),
            self.data_raw.to_vec(),
            Some(self.data_norm.iter().map(|x| x.to_f32()).collect()),
            n_cells,
        )
    }
}

/// SparseDataHeader
///
/// Stores the information in terms of total cells, total genes, number of
/// chunks in terms of cells and genes and the offset vectors
#[derive(Encode, Decode, Serialize, Deserialize, Clone)]
pub struct SparseDataHeader {
    /// Total number of cells in the experiment.
    pub total_cells: usize,
    /// Total number of genes in the experiemnt.
    pub total_genes: usize,
    /// Is the file written in a way for fast cell data retrieval (set to true)
    /// or for fast gene retrieval (set to false)
    pub cell_based: bool,
    /// Number of chunks in this file storing either the cell or gene data
    pub no_chunks: usize,
    /// Offset vector for reading in the data
    pub chunk_offsets: Vec<u64>,
    /// FxHashMap with the original index -> chunk info
    pub index_map: FxHashMap<usize, usize>,
}

/// Fixed-size file header that points to the main header location
///
/// ### Params
///
/// * `magic` - Magic string as bytes to recognise the file
/// * `version` - Version of the file
/// * `main_header_offset` - Offset of the main header, i.e., 64 bytes
/// * `_reserved_1` - 32 additional reserved bytes for the future
/// * `_reserved_2` - 4 additional reserved bytes for the future
#[repr(C)]
#[derive(Encode, Decode, Serialize, Deserialize)]
struct FileHeader {
    magic: [u8; 8],
    version: u32,
    main_header_offset: u64,
    cell_based: bool,
    // Needs to be split into two arrays to get to 64 bytes
    _reserved_1: [u8; 32],
    _reserved_2: [u8; 3],
}

impl FileHeader {
    /// Generate a new header
    ///
    /// ### Params
    ///
    /// * `cell_based` - Is the data stored for fast cell retrieval.
    ///
    /// ### Returns
    ///
    /// A new object of `FileHeader`
    fn new(cell_based: bool) -> Self {
        Self {
            magic: *b"SCRNASEQ",
            version: SC_FILE_VERSION,
            main_header_offset: 0,
            cell_based,
            _reserved_1: [0; 32],
            _reserved_2: [0; 3],
        }
    }
}

//////////////////////
// Streaming writer //
//////////////////////

/// CellGeneSparseWriter
///
/// Implementation of a structure for writing in a streamed manner two different
/// types of sparse stored data.
///
/// ### Params
///
/// * `header` - The header of the file.
/// * `writer` - BufWriter to the file.
/// * `chunks_start_pos` - The current position of the chunks.
/// * `cell_based` - Boolean indicating if the writer is designed to write in
///   an efficient manner for cells.
pub struct CellGeneSparseWriter {
    header: SparseDataHeader,
    writer: BufWriter<File>,
    chunks_start_pos: u64,
    cell_based: bool,
    chunks_since_flush: usize,
    flush_frequency: usize,
}

impl CellGeneSparseWriter {
    /// Create a new sparse writer instance
    ///
    /// This writer assumes that rows represent genes and columns represent
    /// cells.
    ///
    /// ### Params
    ///
    /// * `path_f` - Path to the .bin file to which to write to.
    /// * `cell_based` - Shall the writer be set up for writing cell-based
    ///   (`true`) or gene-based chunks.
    /// * `total_cells` - Total cells in the data.
    /// * `total_genes` - Total genes in the data.
    ///
    /// ### Returns
    ///
    /// The `CellGeneSparseWriter`.
    pub fn new<P: AsRef<Path>>(
        path_f: P,
        cell_based: bool,
        total_cells: usize,
        total_genes: usize,
    ) -> std::io::Result<Self> {
        let file = File::create(path_f)?;
        let mut writer = BufWriter::with_capacity(128 * 1024 * 1024, file);

        let file_header = FileHeader::new(cell_based);
        let file_header_enc = encode_to_vec(&file_header, config::standard()).unwrap();
        if file_header_enc.len() < 64 {
            writer.write_all(&file_header_enc)?;
            writer.write_all(&vec![0u8; 64 - file_header_enc.len()])?;
        } else {
            writer.write_all(&file_header_enc[..64])?;
        }
        writer.flush()?;

        let chunks_start_pos = 64;

        let flush_frequency = if cell_based { 100000_usize } else { 1000_usize };

        let header = SparseDataHeader {
            total_cells,
            total_genes,
            cell_based,
            no_chunks: 0,
            chunk_offsets: Vec::new(),
            index_map: FxHashMap::default(),
        };

        Ok(Self {
            header,
            writer,
            chunks_start_pos,
            cell_based,
            chunks_since_flush: 0_usize,
            flush_frequency,
        })
    }

    /// Write a Cell (Chunk) to the file
    ///
    /// This function will panic if the file was not set to cell-based! The
    /// data is represented in a CSR-type format.
    ///
    /// ### Params
    ///
    /// * `cell_chunk` - The data representing that specific cell.
    pub fn write_cell_chunk(&mut self, cell_chunk: CsrCellChunk) -> std::io::Result<()> {
        assert!(
            self.cell_based,
            "The writer is not set to write in a cell-based manner!"
        );

        let current_pos = self.writer.stream_position()?;
        let chunk_offset = current_pos - self.chunks_start_pos;
        self.header.chunk_offsets.push(chunk_offset);
        self.header
            .index_map
            .insert(cell_chunk.original_index, self.header.no_chunks);

        // serialise to buffer
        let mut buffer = Vec::new();
        cell_chunk.write_to_bytes(&mut buffer)?;

        // compress
        let compressed = compress_prepend_size(&buffer);

        // write compressed size and data
        self.writer
            .write_all(&(compressed.len() as u64).to_le_bytes())?;
        self.writer.write_all(&compressed)?;

        self.header.no_chunks += 1;
        Ok(())
    }

    /// Write a Gene to the file
    ///
    /// This function will panic if the file was set to cell-based!
    ///
    /// ### Params
    ///
    /// * `gene_chunk` - The data representing that specific gene.
    pub fn write_gene_chunk(&mut self, gene_chunk: CscGeneChunk) -> std::io::Result<()> {
        assert!(
            !self.cell_based,
            "The writer is not set to write in a gene-based manner!"
        );

        let current_pos = self.writer.stream_position()?;
        let chunk_offset = current_pos - self.chunks_start_pos;
        self.header.chunk_offsets.push(chunk_offset);
        self.header
            .index_map
            .insert(gene_chunk.original_index, self.header.no_chunks);

        // serialize to buffer
        let mut buffer = Vec::new();
        gene_chunk.write_to_bytes(&mut buffer)?;

        // compress
        let compressed = compress_prepend_size(&buffer);

        // write compressed size and data
        self.writer
            .write_all(&(compressed.len() as u64).to_le_bytes())?;
        self.writer.write_all(&compressed)?;

        self.header.no_chunks += 1;
        self.chunks_since_flush += 1;

        if self.chunks_since_flush >= self.flush_frequency {
            self.writer.flush()?;
            self.chunks_since_flush = 0;
        }

        Ok(())
    }

    /// Finalise the file
    pub fn finalise(mut self) -> std::io::Result<()> {
        // write header size and header
        let main_header_offset = self.writer.stream_position()?;

        let header_data = encode_to_vec(&self.header, config::standard()).unwrap();
        let header_size = header_data.len() as u64;

        self.writer.write_all(&header_size.to_le_bytes())?;
        self.writer.write_all(&header_data)?;

        self.writer.seek(SeekFrom::Start(0))?;
        let mut file_header = FileHeader::new(self.cell_based);
        file_header.main_header_offset = main_header_offset;
        let file_header_enc = encode_to_vec(&file_header, config::standard()).unwrap();

        // ensure it's exactly 64 bytes
        if file_header_enc.len() < 64 {
            self.writer.write_all(&file_header_enc)?;
            self.writer
                .write_all(&vec![0u8; 64 - file_header_enc.len()])?;
        } else {
            self.writer.write_all(&file_header_enc[..64])?;
        }

        self.writer.flush()?;

        Ok(())
    }

    /// Update the number of cells in the header
    ///
    /// Helper function to update the number of cells to the number that was
    /// finally written on disk.
    ///
    /// ### Params
    ///
    /// * `no_cells` - New number of cells
    pub fn update_header_no_cells(&mut self, no_cells: usize) {
        self.header.total_cells = no_cells;
    }

    /// Update the number of cells in the header
    ///
    /// Helper function to update the number of genes to the number that was
    /// finally written on disk.
    ///
    /// ### Params
    ///
    /// * `no_genes` - New number of genes
    pub fn update_header_no_genes(&mut self, no_genes: usize) {
        self.header.total_genes = no_genes;
    }
}

//////////////////////
// Streaming reader //
//////////////////////

/// ParallelSparseReader
///
/// ### Params
///
/// * `header` - The file header
/// * `mmap` - Reference to the memory map for safe sharing across threads
/// * `chunks_start` - Start of the chunks after the hader file
pub struct ParallelSparseReader {
    header: SparseDataHeader,
    mmap: Arc<memmap2::Mmap>,
    chunks_start: u64,
}

impl ParallelSparseReader {
    /// Generate a new parallelised streaming reader
    ///
    /// ### Params
    ///
    /// * `f_path` - Path to the file
    ///
    /// ### Returns
    ///
    /// Initialised `ParallelSparseReader`.
    pub fn new(f_path: &str) -> std::io::Result<Self> {
        let file = File::open(f_path)?;
        let file_size = file.metadata()?.len();

        let mmap = unsafe {
            let mut opts = MmapOptions::new();
            // if the file size is ≤ 8 GB, it loads the full thing into memory
            // otherwise lazy load
            if file_size <= 8 * 1024 * 1024 * 1024 {
                // 8GB
                opts.populate();
            }
            opts.map(&file)?
        };

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;

        // Parse headers from mmap
        let file_header_bytes = &mmap[0..64];
        let (file_header, _) = decode_from_slice::<FileHeader, _>(
            file_header_bytes,
            config::standard(),
        )
        .map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "File header decode failed")
        })?;

        // assert that this is the right file
        assert!(
            file_header.version == SC_FILE_VERSION,
            "File version mismatch: expected {}, got {}. Please check the version you are using",
            SC_FILE_VERSION,
            file_header.version
        );

        // Read main header
        let main_header_offset = file_header.main_header_offset as usize;
        let header_size = u64::from_le_bytes(
            mmap[main_header_offset..main_header_offset + 8]
                .try_into()
                .unwrap(),
        ) as usize;

        let header_bytes = &mmap[main_header_offset + 8..main_header_offset + 8 + header_size];
        let (header, _) =
            decode_from_slice::<SparseDataHeader, _>(header_bytes, config::standard()).map_err(
                |_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Header decode failed"),
            )?;

        Ok(Self {
            header,
            mmap: Arc::new(mmap),
            chunks_start: 64,
        })
    }

    /// Read in cells by indices in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the cells to retrieve
    ///
    /// ### Returns
    ///
    /// Returns an array of `CsrCellChunk`.
    pub fn read_cells_parallel(&self, indices: &[usize]) -> Vec<CsrCellChunk> {
        assert!(
            self.header.cell_based,
            "The file is not set up for CellChunks."
        );

        indices
            .par_iter()
            .map(|&original_index| {
                let chunk_index = *self.header.index_map.get(&original_index).unwrap();
                let chunk_offset =
                    (self.chunks_start + self.header.chunk_offsets[chunk_index]) as usize;

                // read compressed size
                let compressed_size = u64::from_le_bytes(
                    self.mmap[chunk_offset..chunk_offset + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;

                // decompress
                let compressed = &self.mmap[chunk_offset + 8..chunk_offset + 8 + compressed_size];
                let decompressed = decompress_size_prepended(compressed).unwrap();

                CsrCellChunk::read_from_buffer(&decompressed).unwrap()
            })
            .collect()
    }

    /// Read a single cell by index
    ///
    /// ### Params
    ///
    /// * `index` - Cell index
    ///
    /// ### Return
    ///
    /// The CsrCellChunk of this cell
    pub fn read_cell(&self, index: usize) -> CsrCellChunk {
        self.read_cells_parallel(&[index])
            .into_iter()
            .next()
            .unwrap()
    }

    /// Read in genes by indices in a multi-threaded manner
    ///
    /// ### Params
    ///
    /// * `indices` - Slice of index positions of the genes to retrieve
    ///
    /// ### Returns
    ///
    /// Returns an array of `CscGeneChunk`.
    pub fn read_gene_parallel(&self, indices: &[usize]) -> Vec<CscGeneChunk> {
        assert!(
            !self.header.cell_based,
            "The file is not set up for CellChunks."
        );

        indices
            .par_iter()
            .map(|&original_index| {
                let chunk_index = *self.header.index_map.get(&original_index).unwrap();
                let chunk_offset =
                    (self.chunks_start + self.header.chunk_offsets[chunk_index]) as usize;

                // read compressed size
                let compressed_size = u64::from_le_bytes(
                    self.mmap[chunk_offset..chunk_offset + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;

                // decompress
                let compressed = &self.mmap[chunk_offset + 8..chunk_offset + 8 + compressed_size];
                let decompressed = decompress_size_prepended(compressed).unwrap();

                CscGeneChunk::read_from_buffer(&decompressed).unwrap()
            })
            .collect()
    }

    /// Return all cells
    ///
    /// ### Returns
    ///
    /// Returns an array of `CsrCellChunk` containing all cells on disk.
    pub fn get_all_cells(&self) -> Vec<CsrCellChunk> {
        let iter: Vec<usize> = (0..self.header.total_cells).collect();

        self.read_cells_parallel(&iter)
    }

    /// Return all genes
    ///
    /// ### Returns
    ///
    /// Returns an array of `CscGeneChunk` containing all genes on disk.
    pub fn get_all_genes(&self) -> Vec<CscGeneChunk> {
        let iter: Vec<usize> = (0..self.header.total_genes).collect();

        self.read_gene_parallel(&iter)
    }

    /// Helper to return the header
    ///
    /// ### Returns
    ///
    /// Returns the header file
    pub fn get_header(&self) -> SparseDataHeader {
        self.header.clone()
    }

    /// Read cells in a specific range
    ///
    /// Helper for memory-bounded gene generation
    pub fn read_cells_range(&self, start: usize, end: usize) -> Vec<CsrCellChunk> {
        assert!(self.header.cell_based, "File not cell-based");
        let indices: Vec<usize> = (start..end).collect();
        self.read_cells_parallel(&indices)
    }

    /// Read only library sizes for specified cells
    ///
    /// More efficient than reading full chunks when you only need library sizes.
    ///
    /// ### Params
    ///
    /// * `indices` - Cell indices
    ///
    /// ### Returns
    ///
    /// Vector of library sizes
    #[allow(dead_code)]
    pub fn read_cell_library_sizes(&self, indices: &[usize]) -> Vec<usize> {
        assert!(self.header.cell_based, "File not cell-based");

        indices
            .par_iter()
            .map(|&original_index| {
                let chunk_index = *self.header.index_map.get(&original_index).unwrap();
                let chunk_offset =
                    (self.chunks_start + self.header.chunk_offsets[chunk_index]) as usize;

                let compressed_size = u64::from_le_bytes(
                    self.mmap[chunk_offset..chunk_offset + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;

                let compressed = &self.mmap[chunk_offset + 8..chunk_offset + 8 + compressed_size];
                let decompressed = decompress_size_prepended(compressed).unwrap();

                // library size is at bytes 12-19 of the header
                u64::from_le_bytes([
                    decompressed[12],
                    decompressed[13],
                    decompressed[14],
                    decompressed[15],
                    decompressed[16],
                    decompressed[17],
                    decompressed[18],
                    decompressed[19],
                ]) as usize
            })
            .collect()
    }

    /// Read the NNZ for specific genes
    ///
    /// ### Params
    ///
    /// * `indices` - Gene indices
    ///
    /// ### Returns
    ///
    /// Vector of number of NNZ genes.
    pub fn read_gene_nnz(&self, indices: &[usize]) -> Vec<usize> {
        assert!(!self.header.cell_based, "File not gene-based");

        indices
            .par_iter()
            .map(|&original_index| {
                let chunk_index = *self.header.index_map.get(&original_index).unwrap();
                let chunk_offset =
                    (self.chunks_start + self.header.chunk_offsets[chunk_index]) as usize;

                let compressed_size = u64::from_le_bytes(
                    self.mmap[chunk_offset..chunk_offset + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;

                let compressed = &self.mmap[chunk_offset + 8..chunk_offset + 8 + compressed_size];
                let decompressed = decompress_size_prepended(compressed).unwrap();

                // NNZ is at bytes 16-23 of the header
                u64::from_le_bytes([
                    decompressed[16],
                    decompressed[17],
                    decompressed[18],
                    decompressed[19],
                    decompressed[20],
                    decompressed[21],
                    decompressed[22],
                    decompressed[23],
                ]) as usize
            })
            .collect()
    }

    /// Return NNZ for all genes
    ///
    /// ### Returns
    ///
    /// Vector of NNZ values for all genes
    pub fn get_all_gene_nnz(&self) -> Vec<usize> {
        let iter: Vec<usize> = (0..self.header.total_genes).collect();
        self.read_gene_nnz(&iter)
    }
}

//////////////////////
// Chunks to sparse //
//////////////////////

/// Converts a slice of gene chunks into a CSC sparse matrix
///
/// Constructs a cells × genes compressed sparse column matrix from individual
/// gene chunks. The primary data layer contains raw counts, whilst the
/// secondary layer contains normalised counts converted to f32 precision.
///
/// ### Params
///
/// * `chunks` - Slice of gene chunks to convert
/// * `n_cells` - Total number of cells in the dataset
///
/// ### Returns
///
/// A CSC-formatted sparse matrix with raw counts in the primary data layer
/// and normalised counts in the secondary data layer
pub fn from_gene_chunks<T>(chunks: &[CscGeneChunk], n_cells: usize) -> CompressedSparseData2<T, f32>
where
    T: BixverseNumeric + From<u16>,
{
    let n_genes = chunks.len();
    let mut data = Vec::new();
    let mut data_2 = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n_genes + 1);

    indptr.push(0);

    for chunk in chunks {
        for &val in &chunk.data_raw {
            data.push(T::from(val));
        }
        for &val in &chunk.data_norm {
            data_2.push(val.to_f32());
        }
        for &idx in &chunk.indices {
            indices.push(idx as usize);
        }
        indptr.push(data.len());
    }

    CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csc,
        data_2: Some(data_2),
        shape: (n_cells, n_genes),
    }
}

/// Converts a slice of cell chunks into a CSR sparse matrix
///
/// Constructs a cells × genes compressed sparse row matrix from individual
/// cell chunks. The primary data layer contains raw counts, whilst the
/// secondary layer contains normalised counts converted to f32 precision.
///
/// ### Params
///
/// * `chunks` - Slice of cell chunks to convert
/// * `n_genes` - Total number of genes in the dataset
///
/// ### Returns
///
/// A CSR-formatted sparse matrix with raw counts in the primary data layer
/// and normalised counts in the secondary data layer
pub fn from_cell_chunks<T>(chunks: &[CsrCellChunk], n_genes: usize) -> CompressedSparseData2<T, f32>
where
    T: BixverseNumeric + From<u16>,
{
    let n_cells = chunks.len();
    let mut data = Vec::new();
    let mut data_2 = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n_cells + 1);

    indptr.push(0);

    for chunk in chunks {
        for &val in &chunk.data_raw {
            data.push(T::from(val));
        }
        for &val in &chunk.data_norm {
            data_2.push(val.to_f32());
        }
        for &idx in &chunk.indices {
            indices.push(idx as usize);
        }
        indptr.push(data.len());
    }

    CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: Some(data_2),
        shape: (n_cells, n_genes),
    }
}
