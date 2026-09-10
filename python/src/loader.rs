//! The batching loader exposed to Python.
//!
//! [`PyCellLoader`] holds the configuration and the cell index vector.
//! Iterating it spawns one feeder thread that decodes batches ahead of the
//! consumer and pushes them down a bounded channel, so the training step and
//! the decode overlap. The channel depth is the backpressure knob: a full
//! channel blocks the feeder, which is what stops a slow consumer from
//! materialising the whole epoch in memory.
//!
//! One feeder, not several. The reader's `read_cells_parallel` already
//! saturates the rayon pool; a second feeder would nest rayon calls and
//! oversubscribe the cores for no throughput gain. Prefetch depth buys
//! pipelining against Python, not decode parallelism.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use bixverse_rs::errors::BixverseErrors;
use bixverse_rs::single_cell::sc_data::data_io::{ParallelSparseReader, SingleCellReading};
use crossbeam_channel::{Receiver, bounded};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::batch::{Layer, PreparedBatch, adopt_csr, prepare_dense};
use crate::error::BvErr;
use crate::reader::PyCellReader;

///////////////
// Constants //
///////////////

/// Default cells per batch. Matches the minibatch size scVI and friends use by
/// default, and is large enough that per-batch overheads disappear against the
/// decode.
const DEFAULT_BATCH_SIZE: usize = 256;

/// Default number of batches held in flight.
///
/// Four is enough to cover a training step that takes a few times longer than
/// a decode without pinning much memory: at 256 cells and 20k genes a sparse
/// batch is a couple of MB, so the whole queue costs single-digit MB.
const DEFAULT_PREFETCH: usize = 4;

////////////////
// Batch send //
////////////////

/// What the feeder pushes down the channel.
///
/// Carries the crate's own error rather than a `PyErr`, so the feeder never
/// touches Python state off the GIL.
type BatchMsg = Result<PreparedBatch, BixverseErrors>;

/// Convert a materialised batch into the Python-facing tuple.
///
/// The shape is fixed regardless of layer and layout:
/// `(raw, norm, indices, indptr, (n_rows, n_cols))`, with the unrequested
/// slots set to `None`. A uniform shape keeps the type stub honest and saves
/// the caller from branching on what it asked for.
///
/// ### Params
///
/// * `py` - GIL token.
/// * `batch` - Owned batch produced by the feeder thread.
///
/// ### Returns
///
/// The five-tuple described above.
fn batch_to_py(py: Python<'_>, batch: PreparedBatch) -> PyResult<Py<PyAny>> {
    let shape = (batch.n_rows, batch.n_cols);
    let dense = batch.indices.is_none();

    // Dense buffers are one flat row-major block and get reshaped; sparse
    // buffers stay one-dimensional.
    let to_values = |data: Option<Vec<f32>>| -> PyResult<Option<Py<PyAny>>> {
        let Some(data) = data else { return Ok(None) };
        let arr = data.into_pyarray(py);
        if dense {
            Ok(Some(arr.reshape([batch.n_rows, batch.n_cols])?.into()))
        } else {
            Ok(Some(arr.into()))
        }
    };

    let raw = to_values(batch.raw)?;
    let norm = to_values(batch.norm)?;
    let indices: Option<Py<PyAny>> = batch.indices.map(|v| v.into_pyarray(py).into());
    let indptr: Option<Py<PyAny>> = batch.indptr.map(|v| v.into_pyarray(py).into());

    Ok((raw, norm, indices, indptr, shape).into_pyobject(py)?.into())
}

//////////////////
// PyCellLoader //
//////////////////

/// Epoch-wise batching over a cell-major store.
#[pyclass(module = "bixverse._bixverse", name = "CellLoader", frozen)]
pub struct PyCellLoader {
    /// The memory-mapped reader, shared with each feeder thread.
    reader: Arc<ParallelSparseReader>,
    /// Cell indices to iterate, in the caller's order.
    indices: Vec<usize>,
    /// Cells per batch.
    batch_size: usize,
    /// Whether the index vector is reshuffled at the start of each epoch.
    shuffle: bool,
    /// Base seed. `None` draws one from the OS on the first epoch.
    seed: Option<u64>,
    /// Which data layer(s) each batch carries.
    layer: Layer,
    /// `true` for CSR component vectors, `false` for a dense buffer.
    sparse: bool,
    /// Batches held in flight by the feeder.
    prefetch: usize,
    /// Whether a trailing partial batch is dropped.
    drop_last: bool,
    /// Total genes, cached so the feeder does not touch the header.
    n_genes: usize,
    /// Epochs started so far, mixed into the shuffle seed.
    epoch: AtomicU64,
}

#[pymethods]
impl PyCellLoader {
    /// Build a loader over a set of cells.
    ///
    /// ### Params
    ///
    /// * `reader` - An open cell-major [`PyCellReader`].
    /// * `indices` - Cell indices as a contiguous `uint32` array. Validated on
    ///   the Python side, and bounds-checked here once per construction.
    /// * `batch_size` - Cells per batch.
    /// * `shuffle` - Reshuffle at the start of every epoch.
    /// * `seed` - Base seed for the shuffle. `None` draws one at random.
    /// * `layer` - `"raw"`, `"norm"` or `"both"`.
    /// * `sparse` - CSR components when `true`, a dense buffer when `false`.
    /// * `prefetch` - Batches held in flight.
    /// * `drop_last` - Drop a trailing partial batch.
    ///
    /// ### Returns
    ///
    /// The loader, or an exception for a gene-major store, an unusable
    /// parameter, or an out-of-range cell index.
    #[new]
    #[pyo3(signature = (
        reader,
        indices,
        batch_size = DEFAULT_BATCH_SIZE,
        shuffle = true,
        seed = None,
        layer = "norm",
        sparse = true,
        prefetch = DEFAULT_PREFETCH,
        drop_last = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        reader: &PyCellReader,
        indices: PyReadonlyArray1<'_, u32>,
        batch_size: usize,
        shuffle: bool,
        seed: Option<u64>,
        layer: &str,
        sparse: bool,
        prefetch: usize,
        drop_last: bool,
    ) -> PyResult<Self> {
        if !reader.cell_based {
            return Err(BvErr(BixverseErrors::ReaderModeMismatch {
                actual: "gene-based",
                requested: "cell-based",
            })
            .into());
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        if prefetch == 0 {
            return Err(PyValueError::new_err("prefetch must be positive"));
        }

        let layer = Layer::parse(layer).map_err(PyValueError::new_err)?;

        let slice = indices.as_slice()?;
        let total_cells = reader.total_cells;
        if let Some(&bad) = slice.iter().find(|&&i| i as usize >= total_cells) {
            return Err(PyValueError::new_err(format!(
                "cell index {bad} is out of range for a store of {total_cells} cells"
            )));
        }

        Ok(Self {
            reader: Arc::clone(&reader.inner),
            indices: slice.iter().map(|&i| i as usize).collect(),
            batch_size,
            shuffle,
            seed,
            layer,
            sparse,
            prefetch,
            drop_last,
            n_genes: reader.total_genes,
            epoch: AtomicU64::new(0),
        })
    }

    /// Number of batches in one epoch.
    fn __len__(&self) -> usize {
        let n = self.indices.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Start an epoch.
    ///
    /// Each call reshuffles, spawns its own feeder thread and returns a fresh
    /// iterator, so two concurrent passes over one loader do not share state.
    ///
    /// ### Returns
    ///
    /// A [`PyCellLoaderIter`] over this epoch's batches.
    fn __iter__(&self) -> PyCellLoaderIter {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);

        let mut order = self.indices.clone();
        if self.shuffle {
            let base = self.seed.unwrap_or_else(rand::random);
            let mut rng = StdRng::seed_from_u64(base ^ epoch);
            order.shuffle(&mut rng);
        }
        if self.drop_last {
            order.truncate((order.len() / self.batch_size) * self.batch_size);
        }

        let (sender, receiver) = bounded::<BatchMsg>(self.prefetch);
        let reader = Arc::clone(&self.reader);
        let (batch_size, n_genes, layer, sparse) =
            (self.batch_size, self.n_genes, self.layer, self.sparse);

        let handle = std::thread::spawn(move || {
            for batch in order.chunks(batch_size) {
                // The sparse path goes through the crate's fused reader, which
                // walks the decompressed bytes once instead of building a
                // `CsrCellChunk` per cell and copying out of it again.
                let msg = if sparse {
                    reader
                        .read_cells_csr(batch, &layer.as_data_layer())
                        .map(adopt_csr)
                } else {
                    reader
                        .read_cells_parallel(batch)
                        .map(|chunks| prepare_dense(&chunks, n_genes, layer))
                };
                let failed = msg.is_err();
                // A send error means the consumer went away; stop quietly.
                if sender.send(msg).is_err() || failed {
                    break;
                }
            }
        });

        PyCellLoaderIter {
            receiver: Some(receiver),
            handle: Some(handle),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CellLoader(n_cells={}, batch_size={}, layer={:?}, sparse={}, shuffle={})",
            self.indices.len(),
            self.batch_size,
            self.layer,
            self.sparse,
            self.shuffle
        )
    }
}

//////////////////////
// PyCellLoaderIter //
//////////////////////

/// One epoch's worth of batches, backed by a live feeder thread.
#[pyclass(module = "bixverse._bixverse", name = "CellLoaderIter")]
pub struct PyCellLoaderIter {
    /// Receiving end of the prefetch channel. Dropped to signal the feeder.
    receiver: Option<Receiver<BatchMsg>>,
    /// The feeder, joined once the channel closes or the iterator is dropped.
    handle: Option<JoinHandle<()>>,
}

impl PyCellLoaderIter {
    /// Close the channel and wait for the feeder to exit.
    ///
    /// Dropping the receiver first is what unblocks a feeder parked on a full
    /// channel, so the order matters.
    fn shutdown(&mut self) {
        self.receiver = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[pymethods]
impl PyCellLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Pull the next decoded batch, blocking without the GIL.
    ///
    /// ### Returns
    ///
    /// `(raw, norm, indices, indptr, shape)`, or `None` once the epoch ends.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(receiver) = self.receiver.as_ref() else {
            return Ok(None);
        };

        match py.detach(|| receiver.recv()) {
            Ok(Ok(batch)) => batch_to_py(py, batch).map(Some),
            Ok(Err(e)) => {
                self.shutdown();
                Err(BvErr(e).into())
            }
            // Disconnected: the feeder finished the epoch and dropped its end.
            Err(_) => {
                self.shutdown();
                Ok(None)
            }
        }
    }
}

impl Drop for PyCellLoaderIter {
    fn drop(&mut self) {
        self.shutdown();
    }
}
