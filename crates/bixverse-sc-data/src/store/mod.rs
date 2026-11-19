//! The bixverse binary store: a cell-major (CSR) and a gene-major (CSC) file,
//! one compressed chunk per cell or gene. Chunk codecs, the v4 and v5
//! containers, readers, writers and the store-level tools (gene file build,
//! archive, merge, in-memory twin) live here.

pub mod archive;
pub mod chunk;
pub mod convert;
pub mod gene_file;
pub mod in_memory;
pub mod interval_index;
pub mod merge;
pub mod quality;
pub mod reader;
pub mod v4;
pub mod v5;
pub mod writer;

/// How many cells to read in per given batch - global constant
pub const CELL_BATCH_SIZE: usize = 50_000;

/// Version of the single cell files writers emit: the v4 chunk codec behind
/// the v5 container, see [`v5`].
pub const SC_FILE_VERSION: u32 = 5;

/// v4: delta bit-packed chunks with the normalised layer recomputed on read,
/// bincode header. Readers still open it.
pub const SC_FILE_VERSION_V4: u32 = 4;

/// v3: absolute `u32` indices, normalised layer stored. Only the converters
/// read it; [`convert::upgrade_store_to_v5`] converts it.
pub const SC_FILE_VERSION_V3: u32 = 3;
