//! Synthetic store generation, for the Python test suite only.
//!
//! Exposed so `tests/conftest.py` can build a fixture without shelling out to
//! cargo and without reimplementing the crate's RNG on the Python side. The
//! dense truth matrix comes back alongside the file, so a test can assert on
//! exact values rather than on shapes alone.
//!
//! Everything here is prefixed with an underscore on the Python side and is
//! not part of the package's public surface.

use bixverse_rs::single_cell::sc_data::data_io::{CellGeneSparseWriter, CsrCellChunk};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::BvErr;

/// Library size the normalised layer is scaled against in generated stores.
const TARGET_SIZE: f32 = 1e4;

/// Largest raw count drawn, inclusive. Small enough to stay in the `u16`
/// element width so the fixture exercises the common on-disk layout.
const MAX_COUNT: u32 = 20;

/// Write a synthetic cell-major store and return its dense raw counts.
///
/// Every cell is given at least one non-zero, since a wholly empty cell is a
/// different edge case and not what these fixtures are for.
///
/// ### Params
///
/// * `py` - GIL token.
/// * `path` - Where to write the `.bin`.
/// * `n_cells` - Cells to generate.
/// * `n_genes` - Genes per cell.
/// * `density` - Probability that any given gene is non-zero in a cell.
/// * `seed` - RNG seed, so a fixture is reproducible.
///
/// ### Returns
///
/// The dense `(n_cells, n_genes)` raw counts as `f32`, matching what the
/// loader yields for `layer="raw"`.
#[pyfunction]
#[pyo3(name = "_write_synthetic_store")]
#[pyo3(signature = (path, n_cells, n_genes, density = 0.3, seed = 42))]
pub fn write_synthetic_store<'py>(
    py: Python<'py>,
    path: &str,
    n_cells: usize,
    n_genes: usize,
    density: f32,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut dense = vec![0f32; n_cells * n_genes];

    let mut writer =
        CellGeneSparseWriter::new(path, true, n_cells, n_genes, TARGET_SIZE).map_err(BvErr)?;

    for cell in 0..n_cells {
        let mut counts: Vec<u32> = Vec::new();
        let mut cols: Vec<u32> = Vec::new();

        for gene in 0..n_genes {
            if rng.random::<f32>() < density {
                let count = rng.random_range(1..=MAX_COUNT);
                counts.push(count);
                cols.push(gene as u32);
                dense[cell * n_genes + gene] = count as f32;
            }
        }

        if counts.is_empty() {
            counts.push(1);
            cols.push(0);
            dense[cell * n_genes] = 1.0;
        }

        let chunk = CsrCellChunk::from_data(&counts, &cols, cell, TARGET_SIZE, true);
        writer.write_cell_chunk(chunk).map_err(BvErr)?;
    }

    writer.finalise().map_err(BvErr)?;

    Ok(dense.into_pyarray(py).reshape([n_cells, n_genes])?.into_any())
}
