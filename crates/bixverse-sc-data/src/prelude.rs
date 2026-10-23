//! Re-exports for `use crate::prelude::*`: the sparse prelude (plus the core
//! prelude under `analysis`) and the single cell data layer.

pub use bixverse_sparse::prelude::*;

#[cfg(feature = "analysis")]
pub use bixverse_core::prelude::*;

pub use crate::data_io::{
    CellQuality, CscGeneChunk, CsrCellChunk, DataLayerReturn, MinCellQuality, ParallelSparseReader,
    RawCounts, SingleCellReading, from_cell_chunks, from_gene_chunks,
};
pub use crate::errors::*;
pub use crate::sc_traits::*;
pub use crate::{CELL_BATCH_SIZE, SC_FILE_VERSION, SC_FILE_VERSION_V3};
