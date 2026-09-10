//! Batch materialisation for the single-cell data loader.
//!
//! Takes decompressed [`CsrCellChunk`]s coming out of the reader and packs
//! them into numpy-friendly buffers: either the three component vectors of a
//! CSR matrix (`data`, `indices`, `indptr`), or a dense row-major
//! `(n_rows, n_genes)` buffer.
//!
//! Sparse is the default and the cheap path. Dense costs a zero-fill and a
//! scatter over `n_rows * n_genes` floats regardless of how few non-zeros the
//! batch actually holds, which at typical droplet sparsity is roughly twenty
//! times the memory traffic for the same information.
//!
//! The work is sequential on purpose. The reader already parallelises
//! decompression across rayon workers; wrapping this pass in another rayon
//! call would oversubscribe the pool with no throughput win.

use bixverse_rs::single_cell::sc_data::data_io::{CsrBatch, CsrCellChunk, DataLayerReturn};

//////////////////
// Layer select //
//////////////////

/// Which data layer(s) to materialise from each chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Raw counts (u16 or u32 on disk), cast to f32.
    Raw,
    /// Normalised counts, stored as f16 on disk and expanded to f32.
    Norm,
    /// Both layers, sharing one sparsity pattern.
    Both,
}

impl Layer {
    /// Parse a layer from its string tag.
    ///
    /// ### Params
    ///
    /// * `s` - One of `"raw"`, `"norm"`, `"both"` (case-insensitive).
    ///
    /// ### Returns
    ///
    /// The matching [`Layer`], or an `Err` naming the accepted values.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "raw" => Ok(Layer::Raw),
            "norm" => Ok(Layer::Norm),
            "both" => Ok(Layer::Both),
            other => Err(format!(
                "unknown layer '{other}': expected one of 'raw', 'norm', 'both'"
            )),
        }
    }

    /// Whether the raw layer is requested.
    ///
    /// ### Returns
    ///
    /// `true` for [`Layer::Raw`] and [`Layer::Both`].
    fn wants_raw(self) -> bool {
        matches!(self, Layer::Raw | Layer::Both)
    }

    /// Whether the normalised layer is requested.
    ///
    /// ### Returns
    ///
    /// `true` for [`Layer::Norm`] and [`Layer::Both`].
    fn wants_norm(self) -> bool {
        matches!(self, Layer::Norm | Layer::Both)
    }

    /// The crate's equivalent layer selector.
    ///
    /// ### Returns
    ///
    /// The matching [`DataLayerReturn`].
    pub fn as_data_layer(self) -> DataLayerReturn {
        match self {
            Layer::Raw => DataLayerReturn::Raw,
            Layer::Norm => DataLayerReturn::Norm,
            Layer::Both => DataLayerReturn::BothLayers,
        }
    }
}

///////////////////
// PreparedBatch //
///////////////////

/// Owned, numpy-ready representation of one materialised batch.
///
/// `raw` and `norm` are populated according to the requested [`Layer`]; at
/// least one of them is always `Some`. `indices` and `indptr` are `Some` on
/// the sparse path and `None` on the dense one, and are shared across both
/// layers because raw and norm have an identical sparsity pattern by
/// construction.
pub struct PreparedBatch {
    /// Raw counts as f32, CSR-ordered or dense row-major.
    pub raw: Option<Vec<f32>>,
    /// Normalised values as f32, in the same layout as `raw`.
    pub norm: Option<Vec<f32>>,
    /// Column (gene) indices, one per non-zero. `None` when dense.
    pub indices: Option<Vec<u32>>,
    /// Row pointers of length `n_rows + 1`. `None` when dense.
    pub indptr: Option<Vec<u32>>,
    /// Number of cells in this batch.
    pub n_rows: usize,
    /// Number of genes, i.e. the CSR width or the dense column stride.
    pub n_cols: usize,
}

//////////////////
// Materialiser //
//////////////////

/// Adopt a [`CsrBatch`] the crate's fused reader already materialised.
///
/// Nothing is copied here: the reader wrote the buffers in the layout numpy
/// wants, so this only moves them into the shape the bindings return.
///
/// ### Params
///
/// * `batch` - The batch as built by `ParallelSparseReader::read_cells_csr`.
///
/// ### Returns
///
/// The equivalent [`PreparedBatch`].
pub fn adopt_csr(batch: CsrBatch) -> PreparedBatch {
    PreparedBatch {
        raw: batch.data_raw,
        norm: batch.data_norm,
        indices: Some(batch.indices),
        indptr: Some(batch.indptr),
        n_rows: batch.shape.0,
        n_cols: batch.shape.1,
    }
}

/// Scatter the chunks into a zero-filled dense row-major buffer.
///
/// ### Params
///
/// * `chunks` - Cell chunks in row order.
/// * `n_genes` - Column stride and total gene count.
/// * `layer` - Which data layer(s) to emit.
///
/// ### Returns
///
/// A [`PreparedBatch`] with `indices` and `indptr` left `None`.
pub fn prepare_dense(chunks: &[CsrCellChunk], n_genes: usize, layer: Layer) -> PreparedBatch {
    let n_rows = chunks.len();
    let total = n_rows.saturating_mul(n_genes);

    let mut raw = layer.wants_raw().then(|| vec![0f32; total]);
    let mut norm = layer.wants_norm().then(|| vec![0f32; total]);

    for (row, chunk) in chunks.iter().enumerate() {
        let row_start = row * n_genes;
        if let Some(buf) = raw.as_mut() {
            for (value, &col) in chunk.data_raw.iter().zip(chunk.indices.iter()) {
                buf[row_start + col as usize] = value as f32;
            }
        }
        if let Some(buf) = norm.as_mut() {
            for (value, &col) in chunk.data_norm.iter().zip(chunk.indices.iter()) {
                buf[row_start + col as usize] = value.to_f32();
            }
        }
    }

    PreparedBatch {
        raw,
        norm,
        indices: None,
        indptr: None,
        n_rows,
        n_cols: n_genes,
    }
}
