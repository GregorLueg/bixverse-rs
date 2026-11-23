//! Re-exports for `use crate::prelude::*`: the sparse prelude (plus the core
//! prelude under `analysis`) and the single cell data layer.

pub use bixverse_sparse::prelude::*;

#[cfg(feature = "analysis")]
pub use bixverse_core::prelude::*;

pub use crate::errors::*;
pub use crate::sc_traits::*;
pub use crate::store::{CELL_BATCH_SIZE, SC_FILE_VERSION, SC_FILE_VERSION_V3, SC_FILE_VERSION_V4};
pub use crate::store::{
    chunk::{
        CscGeneChunk, CsrCellChunk, DataLayerReturn, RawCounts, from_cell_chunks, from_gene_chunks,
    },
    quality::{CellQuality, MinCellQuality},
    reader::{ParallelSparseReader, SingleCellReading},
};
