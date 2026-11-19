//! Writing a store, or an in-memory CSR matrix, to h5ad or AnnData Zarr.
//!
//! The counts stream out of a cell-based store batch by batch, so `X` never
//! sits in memory whole. Everything besides `X` ([`AdParts`]) is checked
//! against the matrix shape before the output is created, so bad input fails
//! before the counts are streamed, not after.

use bixverse_sc_data::store::{
    chunk::DataLayerReturn,
    reader::{ParallelSparseReader, SingleCellReading},
};
#[cfg(feature = "zarr")]
use scx_core::api::write::ZarrAdBuilder;
use scx_core::api::write::{H5AdBuilder, H5AdOptions};
use std::path::Path;

pub use scx_core::ir::{
    Column, ColumnData, DenseMatrix, Embeddings, LevelType, ObsTable, UnsTable, VarTable, Varm,
};
pub use sprs::CsMatI;

use crate::errors::ExportErrors;

///////////
// Types //
///////////

/// The output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdFormat {
    /// An `.h5ad` file.
    H5ad {
        /// gzip level `0..=9` of the numeric datasets, `None` uncompressed.
        compression: Option<u8>,
    },
    /// An AnnData Zarr v3 store (a directory).
    #[cfg(feature = "zarr")]
    Zarr,
}

/// Everything an AnnData object holds besides `X`.
#[derive(Debug, Default)]
pub struct AdParts {
    /// Cell table, one row per written cell.
    pub obs: ObsTable,
    /// Gene table, one row per gene of the store.
    pub var: VarTable,
    /// Cell embeddings, one row per cell.
    pub obsm: Embeddings,
    /// Gene loadings, one row per gene.
    pub varm: Varm,
    /// Cell x cell graphs.
    pub obsp: Vec<(String, CsMatI<f32, u32>)>,
    /// Unstructured metadata.
    pub uns: UnsTable,
}

impl AdParts {
    /// Error unless every part fits an `n_obs x n_vars` matrix.
    ///
    /// ### Params
    ///
    /// * `n_obs` - Cells
    /// * `n_vars` - Genes
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ExportErrors::InvalidArgument`] naming the first part
    /// that does not fit.
    pub fn check(&self, n_obs: usize, n_vars: usize) -> Result<(), ExportErrors> {
        let fits = |what: String, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(ExportErrors::InvalidArgument(format!(
                    "{what} has {got} rows, expected {want}"
                )))
            }
        };
        for (frame, index, cols, n) in [
            ("obs", &self.obs.index, &self.obs.columns, n_obs),
            ("var", &self.var.index, &self.var.columns, n_vars),
        ] {
            fits(format!("{frame} index"), index.len(), n)?;
            for c in cols {
                fits(format!("{frame} column '{}'", c.name), c.data.len(), n)?;
            }
        }
        for (slot, map, n) in [
            ("obsm", &self.obsm.map, n_obs),
            ("varm", &self.varm.map, n_vars),
        ] {
            for (name, m) in map {
                fits(format!("{slot} '{name}'"), m.shape.0, n)?;
                if m.data.len() != m.shape.0 * m.shape.1 {
                    return Err(ExportErrors::InvalidArgument(format!(
                        "{slot} '{name}' holds {} values for shape {:?}",
                        m.data.len(),
                        m.shape
                    )));
                }
            }
        }
        for (name, m) in &self.obsp {
            if m.shape() != (n_obs, n_obs) {
                return Err(ExportErrors::InvalidArgument(format!(
                    "obsp '{name}' is {:?}, expected ({n_obs}, {n_obs})",
                    m.shape()
                )));
            }
        }
        Ok(())
    }
}

/// One of the scx builders.
enum Builder {
    H5(Box<H5AdBuilder>),
    #[cfg(feature = "zarr")]
    Zarr(Box<ZarrAdBuilder>),
}

/// Call the same method on whichever builder it is.
macro_rules! each {
    ($b:expr, $m:ident($($a:expr),*)) => {
        match $b {
            Builder::H5(b) => b.$m($($a),*).map(|_| ()),
            #[cfg(feature = "zarr")]
            Builder::Zarr(b) => b.$m($($a),*).map(|_| ()),
        }
    };
}

impl Builder {
    /// Create the output.
    fn new(
        path: &Path,
        format: AdFormat,
        n_obs: usize,
        n_vars: usize,
        chunk_size: usize,
    ) -> Result<Self, ExportErrors> {
        Ok(match format {
            AdFormat::H5ad { compression } => {
                let mut opts = H5AdOptions::default();
                opts.compression = compression;
                opts.chunk_size = Some(chunk_size);
                Self::H5(Box::new(H5AdBuilder::new(path, n_obs, n_vars, &opts)?))
            }
            #[cfg(feature = "zarr")]
            AdFormat::Zarr => Self::Zarr(Box::new(ZarrAdBuilder::new(
                path,
                n_obs,
                n_vars,
                Some(chunk_size),
            )?)),
        })
    }

    /// Write everything besides `X` and close the output.
    fn finish(mut self, parts: AdParts) -> Result<(), ExportErrors> {
        each!(&mut self, obs(parts.obs))?;
        each!(&mut self, var(parts.var))?;
        each!(&mut self, add_obsm(parts.obsm))?;
        each!(&mut self, add_varm(parts.varm))?;
        for (name, m) in &parts.obsp {
            each!(&mut self, add_obsp_csr(name, m.view()))?;
        }
        each!(&mut self, add_uns(parts.uns))?;
        each!(self, finalize())?;
        Ok(())
    }
}

///////////////
// Functions //
///////////////

/// Write cells of a store to h5ad or AnnData Zarr, streaming the counts
///
/// ### Params
///
/// * `reader` - Reader over the cell-based file
/// * `path` - Output to create
/// * `format` - h5ad or Zarr
/// * `cells` - 0-based cells to write, in output order
/// * `norm` - Write the normalised instead of the raw counts
/// * `parts` - obs (one row per entry of `cells`), var (one row per gene)
///   and the rest
/// * `chunk_size` - Cells per streamed batch
///
/// ### Returns
///
/// `Ok(())`, or the first error. Nothing is created when the inputs do not
/// fit together.
pub fn save_store(
    reader: &ParallelSparseReader,
    path: &Path,
    format: AdFormat,
    cells: &[usize],
    norm: bool,
    parts: AdParts,
    chunk_size: usize,
) -> Result<(), ExportErrors> {
    let dims = reader.dims();
    if !dims.cell_based {
        return Err(ExportErrors::InvalidArgument(
            "export reads the cell-based file, got the gene-based one".into(),
        ));
    }
    if let Some(&c) = cells.iter().find(|&&c| c >= dims.total_cells) {
        return Err(ExportErrors::InvalidArgument(format!(
            "cell {c} is outside the store's {} cells",
            dims.total_cells
        )));
    }
    let (n_obs, n_vars) = (cells.len(), dims.total_genes);
    parts.check(n_obs, n_vars)?;
    let chunk_size = chunk_size.max(1);
    let layer = if norm {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };

    let mut builder = Builder::new(path, format, n_obs, n_vars, chunk_size)?;
    for (i, batch) in cells.chunks(chunk_size).enumerate() {
        let csr = reader.read_cells_csr(batch, &layer)?;
        let data = if norm { csr.data_norm } else { csr.data_raw }
            .expect("the requested layer is always materialised");
        let indptr: Vec<u64> = csr.indptr.iter().map(|&p| p as u64).collect();
        each!(
            &mut builder,
            push_x_csr_chunk(i * chunk_size, indptr, csr.indices, data)
        )?;
    }
    builder.finish(parts)
}

/// Write an in-memory cells x genes CSR matrix to h5ad or AnnData Zarr
///
/// ### Params
///
/// * `path` - Output to create
/// * `format` - h5ad or Zarr
/// * `indptr` - Row pointers over cells, `obs` rows + 1 long
/// * `indices` - 0-based gene per non-zero
/// * `data` - Values, one per non-zero
/// * `parts` - obs, var and the rest; their lengths set the shape
///
/// ### Returns
///
/// `Ok(())`, or an error for inconsistent input before anything is created.
pub fn write_csr(
    path: &Path,
    format: AdFormat,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    data: Vec<f32>,
    parts: AdParts,
) -> Result<(), ExportErrors> {
    let (n_obs, n_vars) = (parts.obs.index.len(), parts.var.index.len());
    parts.check(n_obs, n_vars)?;
    if indptr.len() != n_obs + 1 {
        return Err(ExportErrors::InvalidArgument(format!(
            "indptr has length {}, expected {}",
            indptr.len(),
            n_obs + 1
        )));
    }
    let nnz = indptr.last().copied().unwrap_or(0) as usize;
    if indices.len() != nnz || data.len() != nnz {
        return Err(ExportErrors::InvalidArgument(format!(
            "indptr ends at {nnz} with {} indices and {} values",
            indices.len(),
            data.len()
        )));
    }
    if indices.iter().any(|&j| j as usize >= n_vars) {
        return Err(ExportErrors::InvalidArgument(
            "indices hold a column outside var".into(),
        ));
    }
    let mut builder = Builder::new(path, format, n_obs, n_vars, n_obs.max(1))?;
    each!(&mut builder, push_x_csr_chunk(0, indptr, indices, data))?;
    builder.finish(parts)
}
