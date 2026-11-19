//! Cell and gene quality structures that decide what is read from and
//! written to a store.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;

//////////////////
// Cell Quality //
//////////////////

/// Structure to store QC information on cells
#[derive(Clone, Debug)]
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
///
/// Deserialises from the keys of the R `params_sc_min_quality()` list; every
/// key is required.
#[derive(Clone, Debug, Deserialize)]
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
    /// Stored entries per kept cell over the kept genes, in new-index order.
    /// Only the CSC h5ad parse fills it; empty otherwise.
    pub cell_nnz: Vec<usize>,
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
            cell_nnz: Vec::new(),
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
