//! Re-exports for `use bixverse_sparse::prelude::*`.

pub use crate::errors::SparseErrors;
pub use crate::sparse::{
    CompressedSparseData2, CompressedSparseFormat, SparseAxis, parse_compressed_sparse_format,
    thread_chunks,
};
pub use crate::traits::{BixverseNumeric, VecIndexCast};
