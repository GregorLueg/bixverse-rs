//! Ingestion and store maintenance, the Python twin of the extendr
//! `SingleCellCountData` methods and the obs / var Parquet exporters.
//!
//! Every function is a thin call into `bixverse-sc-data`: arguments come in
//! as paths, numpy arrays and plain dicts, parameter bundles are deserialised
//! with `pythonize` straight into the crate's structs, and the work runs with
//! the GIL released. Indices cross the boundary 0-based in both directions.

use std::path::{Path, PathBuf};

use bixverse_sc_data::archive_io::{archive_cell_file, restore_archive};
use bixverse_sc_data::bin_merge_io::{BinMergeTask, merge_sc_bin_files};
use bixverse_sc_data::data_io::{
    CellQuality, MinCellQuality, ParallelSparseReader, SingleCellReading, peek_file_version,
};
use bixverse_sc_data::gene_file_io::{GENE_FILE_BYTES_PER_NNZ, write_gene_file};
use bixverse_sc_data::h5_10x_io::{
    TenxFeatures, TenxVersion, detect_tenx_version, get_tenx_dimensions, parse_tenx_version,
    stream_h5_tenx_counts, write_tenx_h5,
};
use bixverse_sc_data::h5_10x_multifile_io::{TenxFileTask, multi_10x_h5_to_file};
use bixverse_sc_data::h5_scan::{detect_raw_count_slot, get_h5ad_dimensions, read_h5_strings};
use bixverse_sc_data::h5ad_io::{
    RawDataSlot, parse_raw_slot, stream_h5_counts, write_h5_counts, write_h5_normalised_counts,
};
use bixverse_sc_data::h5ad_multifile_io::{self, H5adFileTask};
use bixverse_sc_data::mtx_io::MtxReader;
use bixverse_sc_data::mtx_multifile_io::{self, MtxFileTask};
use bixverse_sc_data::obs_var_io::{
    FrameKind, ParquetColumnType, ParquetFrameInfo, h5ad_frame_to_parquet, parse_frame_kind,
    tenx_frame_to_parquet, to_snake_case,
};
use bixverse_sc_data::prelude::CompressedSparseData2;
use bixverse_sc_data::r_obj_io::write_r_counts;
use bixverse_sc_data::v4_io::{parse_chunk_codec, upgrade_store_to_v4};
use numpy::{IntoPyArray, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use serde::de::DeserializeOwned;

use crate::error::BvErr;

/////////////
// Helpers //
/////////////

/// A path as UTF-8, which the crate's string-typed signatures need
///
/// ### Params
///
/// * `p` - The path
///
/// ### Returns
///
/// The path as `&str`, or a `ValueError` for a non UTF-8 path.
fn path_str(p: &Path) -> PyResult<&str> {
    p.to_str()
        .ok_or_else(|| PyValueError::new_err(format!("{} is not valid UTF-8", p.display())))
}

/// Deserialise a parameter dict or task list into a crate struct
///
/// ### Params
///
/// * `obj` - The Python object, as `to_rust(params)` or a prescan builds it
/// * `what` - What it is, for the error message
///
/// ### Returns
///
/// The struct, or a `ValueError` naming the offending key.
fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    depythonize(obj).map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
}

/// Memory budget of the CSR to CSC conversion as a non-zero count
///
/// ### Params
///
/// * `max_mem_gb` - Budget in GB, `None` for a single phase
///
/// ### Returns
///
/// The maximum non-zeros held at once, at least one.
fn max_nnz(max_mem_gb: Option<f64>) -> Option<usize> {
    max_mem_gb.map(|gb| ((gb * 1e9) as usize / GENE_FILE_BYTES_PER_NNZ).max(1))
}

/// Directory for the mtx loader's spill files
///
/// The spill is 12 bytes per entry, so it belongs next to the store rather
/// than in the system temp directory, which can sit on a small volume.
///
/// ### Params
///
/// * `f_path_cells` - Path of the cell-based binary being written
///
/// ### Returns
///
/// Its parent directory, or `.` for a bare file name.
fn spill_dir(f_path_cells: &Path) -> PathBuf {
    match f_path_cells.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// `usize` values as a numpy `uint32` array
///
/// Cell and gene counts of a store fit `u32` by construction: its pointers
/// are `u32`.
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values
///
/// ### Returns
///
/// The array.
fn u32_array<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|&x| x as u32)
        .collect::<Vec<u32>>()
        .into_pyarray(py)
        .into_any()
}

/// `usize` values as a numpy `uint64` array
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The values
///
/// ### Returns
///
/// The array.
fn u64_array<'py>(py: Python<'py>, v: &[usize]) -> Bound<'py, PyAny> {
    v.iter()
        .map(|&x| x as u64)
        .collect::<Vec<u64>>()
        .into_pyarray(py)
        .into_any()
}

/// The QC result of a single-file load as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `n_cells` - Cells written
/// * `n_genes` - Genes written
/// * `qc` - The [CellQuality]
///
/// ### Returns
///
/// `cell_indices`, `gene_indices` (`uint32`, 0-based, surviving QC),
/// `lib_size`, `nnz` (`uint64`), `n_cells` and `n_genes`.
fn qc_dict<'py>(
    py: Python<'py>,
    n_cells: usize,
    n_genes: usize,
    qc: &CellQuality,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("cell_indices", u32_array(py, &qc.cell_indices))?;
    d.set_item("gene_indices", u32_array(py, &qc.gene_indices))?;
    d.set_item("lib_size", u64_array(py, &qc.lib_size))?;
    d.set_item("nnz", u64_array(py, &qc.nnz))?;
    d.set_item("n_cells", n_cells)?;
    d.set_item("n_genes", n_genes)?;
    Ok(d)
}

/// The per-file result of a multi-file load as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `exp_id` - Experiment id
/// * `cell_indices` - File-local 0-based kept cells, `None` for a merge
/// * `lib_size` - Library sizes of the kept cells
/// * `nnz` - Non-zeros of the kept cells
///
/// ### Returns
///
/// `exp_id`, `cell_indices`, `lib_size` and `nnz`.
fn per_file_dict<'py>(
    py: Python<'py>,
    exp_id: &str,
    cell_indices: Option<&[usize]>,
    lib_size: &[usize],
    nnz: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("exp_id", exp_id)?;
    if let Some(c) = cell_indices {
        d.set_item("cell_indices", u32_array(py, c))?;
    }
    d.set_item("lib_size", u64_array(py, lib_size))?;
    d.set_item("nnz", u64_array(py, nnz))?;
    Ok(d)
}

/// The result of a multi-file load as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `global_gene_indices` - Universe indices of the kept genes
/// * `total_cells` - Cells written
/// * `total_genes` - Genes written
/// * `per_file` - One dict per input, see [per_file_dict]
///
/// ### Returns
///
/// `global_gene_indices`, `total_cells`, `total_genes` and `per_file`.
fn multi_dict<'py>(
    py: Python<'py>,
    global_gene_indices: &[usize],
    total_cells: usize,
    total_genes: usize,
    per_file: Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("global_gene_indices", u32_array(py, global_gene_indices))?;
    d.set_item("total_cells", total_cells)?;
    d.set_item("total_genes", total_genes)?;
    d.set_item("per_file", PyList::new(py, per_file)?)?;
    Ok(d)
}

/// Describe a written Parquet table for Python
///
/// ### Params
///
/// * `py` - GIL token
/// * `info` - The [ParquetFrameInfo] of the written file
///
/// ### Returns
///
/// `names`, `types`, `categories` (a list of level lists or `None`),
/// `skipped` and `n_rows`, the fields of `ParquetFrameInfo` on the Python
/// side.
fn info_dict<'py>(py: Python<'py>, info: ParquetFrameInfo) -> PyResult<Bound<'py, PyDict>> {
    let names: Vec<String> = info.columns.iter().map(|c| c.name.clone()).collect();
    let types: Vec<&str> = info
        .columns
        .iter()
        .map(|c| match c.col_type {
            ParquetColumnType::Integer => "integer",
            ParquetColumnType::Double => "double",
            ParquetColumnType::Boolean => "boolean",
            ParquetColumnType::String => "string",
            ParquetColumnType::Categorical => "categorical",
        })
        .collect();
    let categories: Vec<Option<Vec<String>>> =
        info.columns.into_iter().map(|c| c.categories).collect();

    let d = PyDict::new(py);
    d.set_item("names", names)?;
    d.set_item("types", types)?;
    d.set_item("categories", categories)?;
    d.set_item("skipped", info.skipped)?;
    d.set_item("n_rows", info.n_rows)?;
    Ok(d)
}

/// Parse the frame argument of the Parquet exporters
///
/// ### Params
///
/// * `frame` - `"obs"` or `"var"`
///
/// ### Returns
///
/// The [FrameKind], or a `ValueError` naming the bad value.
fn frame_kind(frame: &str) -> PyResult<FrameKind> {
    parse_frame_kind(frame).ok_or_else(|| {
        PyValueError::new_err(format!(
            "frame must be one of 'obs' or 'var', got '{frame}'"
        ))
    })
}

/// Parse a 10x version string
///
/// ### Params
///
/// * `version` - `"v2"` or `"v3"`
///
/// ### Returns
///
/// The [TenxVersion], or a `ValueError` naming the bad value.
fn tenx_version(version: &str) -> PyResult<TenxVersion> {
    parse_tenx_version(version).ok_or_else(|| {
        PyValueError::new_err(format!("version must be 'v2' or 'v3', got '{version}'"))
    })
}

/// The name of a 10x version as the R side spells it
///
/// ### Params
///
/// * `v` - The version
///
/// ### Returns
///
/// `"v2"` or `"v3"`.
fn tenx_name(v: TenxVersion) -> &'static str {
    match v {
        TenxVersion::V2 => "v2",
        TenxVersion::V3 => "v3",
    }
}

/// Parse a raw count slot name
///
/// ### Params
///
/// * `slot` - `"X"`, `"raw.X"` or `"layers.counts"`
///
/// ### Returns
///
/// The [RawDataSlot], or a `ValueError` naming the bad value.
fn raw_slot(slot: &str) -> PyResult<RawDataSlot> {
    parse_raw_slot(slot).ok_or_else(|| {
        PyValueError::new_err(format!(
            "slot must be one of 'X', 'raw.X' or 'layers.counts', got '{slot}'"
        ))
    })
}

/// The name of a raw count slot as the R side spells it
///
/// ### Params
///
/// * `slot` - The slot
///
/// ### Returns
///
/// `"X"`, `"raw.X"` or `"layers.counts"`.
fn slot_name(slot: RawDataSlot) -> &'static str {
    match slot {
        RawDataSlot::DataX => "X",
        RawDataSlot::RawX => "raw.X",
        RawDataSlot::LayerCounts => "layers.counts",
    }
}

/// Optional 0-based row selection as `usize`
///
/// ### Params
///
/// * `keep` - Optional `uint32` array
///
/// ### Returns
///
/// The indices, or `None`.
fn keep_rows(keep: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Option<Vec<usize>>> {
    keep.map(|k| Ok(k.as_slice()?.iter().map(|&x| x as usize).collect()))
        .transpose()
}

///////////////////
// Single inputs //
///////////////////

/// Write an in-memory CSR count matrix to the cell-based binary
///
/// The twin of `r_data_to_file`: per-cell QC, then the cell file.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `data` - Integral counts, one per non-zero
/// * `indices` - 0-based gene index per non-zero
/// * `indptr` - Row pointers, one more than there are cells
/// * `shape` - `(n_cells, n_genes)`
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The QC dict, see [qc_dict].
#[pyfunction]
#[pyo3(signature = (f_path_cells, data, indices, indptr, shape, qc_params, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn csr_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    data: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    qc_params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    if indptr.len() != shape.0 + 1 {
        return Err(PyValueError::new_err(
            "indptr must have n_cells + 1 entries",
        ));
    }
    let csr = CompressedSparseData2::<u32>::new_csr(
        data.as_slice()?,
        indices.as_slice()?,
        indptr.as_slice()?,
        None,
        shape,
    );
    let (n_cells, n_genes, cell_qc) = py
        .detach(|| write_r_counts(&f_path_cells, csr, qc, verbose))
        .map_err(BvErr::from)?;
    qc_dict(py, n_cells, n_genes, &cell_qc)
}

/// Write the counts of an h5ad file to the cell-based binary
///
/// The twin of `h5ad_to_file` and `h5ad_to_file_streaming`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `h5_path` - The h5ad file
/// * `cs_type` - `"CSR"`, `"CSC"`, `"DENSE_ROW"` or `"DENSE_COL"`
/// * `no_cells` - Rows of obs
/// * `no_genes` - Rows of var
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `slot` - `"X"`, `"raw.X"` or `"layers.counts"`
/// * `streaming` - Stream the counts instead of materialising the filtered
///   matrix first
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The QC dict, see [qc_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, h5_path, cs_type, no_cells, no_genes, qc_params, slot = "X",
    streaming = true, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn h5ad_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    h5_path: PathBuf,
    cs_type: &str,
    no_cells: usize,
    no_genes: usize,
    qc_params: &Bound<'py, PyAny>,
    slot: &str,
    streaming: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let slot = raw_slot(slot)?;
    let cs_type = cs_type.to_string();
    let (n_cells, n_genes, cell_qc) = py
        .detach(|| {
            let run = if streaming {
                stream_h5_counts
            } else {
                write_h5_counts
            };
            run(
                h5_path.as_path(),
                f_path_cells.as_path(),
                &cs_type,
                no_cells,
                no_genes,
                qc,
                slot,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    qc_dict(py, n_cells, n_genes, &cell_qc)
}

/// Write an h5ad holding only normalised counts to the cell-based binary
///
/// The twin of `norm_h5ad_to_file`. Raw counts are reconstructed from
/// `log1p(x / lib_size * target_size)` with the library sizes in an obs
/// column.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `h5_path` - The h5ad file
/// * `cs_type` - `"CSR"` or `"CSC"`
/// * `no_cells` - Rows of obs
/// * `no_genes` - Rows of var
/// * `obs_lib_size_col` - obs column with the library sizes
/// * `target_size` - Target size of the original normalisation
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The QC dict, see [qc_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, h5_path, cs_type, no_cells, no_genes, obs_lib_size_col, target_size,
    qc_params, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn norm_h5ad_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    h5_path: PathBuf,
    cs_type: &str,
    no_cells: usize,
    no_genes: usize,
    obs_lib_size_col: &str,
    target_size: f32,
    qc_params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let cs_type = cs_type.to_string();
    let lib_col = obs_lib_size_col.to_string();
    let (n_cells, n_genes, cell_qc) = py
        .detach(|| {
            write_h5_normalised_counts(
                h5_path.as_path(),
                f_path_cells.as_path(),
                &cs_type,
                no_cells,
                no_genes,
                &lib_col,
                target_size,
                qc,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    qc_dict(py, n_cells, n_genes, &cell_qc)
}

/// Write the counts of an mtx file to the cell-based binary
///
/// The twin of `mtx_to_file` and `mtx_to_file_streaming`. The parsed entries
/// spill next to the store.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `mtx_path` - The `.mtx` or `.mtx.gz` file
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `cells_as_rows` - Whether cells are the rows of the file
/// * `streaming` - Use the streaming writer
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The QC dict, see [qc_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, mtx_path, qc_params, cells_as_rows, streaming = true, verbose = false
))]
pub fn mtx_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    mtx_path: PathBuf,
    qc_params: &Bound<'py, PyAny>,
    cells_as_rows: bool,
    streaming: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let cells = path_str(&f_path_cells)?;
    let res = py
        .detach(|| {
            let mut reader = MtxReader::new(&mtx_path, qc, cells_as_rows)?
                .with_temp_dir(spill_dir(&f_path_cells));
            let quality = reader.parse_mtx_quality(verbose)?;
            if streaming {
                reader.process_mtx_and_write_bin_streaming(cells, &quality, verbose)
            } else {
                reader.process_mtx_and_write_bin(cells, &quality, verbose)
            }
        })
        .map_err(BvErr::from)?;
    qc_dict(py, res.no_cells, res.no_genes, &res.cell_qc)
}

/// Write the gene expression counts of a 10x h5 file to the cell-based binary
///
/// The twin of `tenx_h5_to_file_streaming`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `h5_path` - The 10x h5 file
/// * `version` - `"v2"`, `"v3"`, or `None` to detect it
/// * `no_cells` - Columns of the matrix
/// * `no_genes` - Rows of the matrix, every modality included
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `feature_type` - Modality to keep (v3); `None` is `"Gene Expression"`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The QC dict, see [qc_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, h5_path, version, no_cells, no_genes, qc_params, feature_type = None,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn tenx_h5_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    h5_path: PathBuf,
    version: Option<&str>,
    no_cells: usize,
    no_genes: usize,
    qc_params: &Bound<'py, PyAny>,
    feature_type: Option<String>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let version = version.map(tenx_version).transpose()?;
    let (n_cells, n_genes, cell_qc) = py
        .detach(|| {
            stream_h5_tenx_counts(
                h5_path.as_path(),
                f_path_cells.as_path(),
                version,
                no_cells,
                no_genes,
                qc,
                feature_type.as_deref(),
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    qc_dict(py, n_cells, n_genes, &cell_qc)
}

//////////////////
// Multi inputs //
//////////////////

/// Load several h5ad files into one cell-based binary
///
/// The twin of `multi_h5ad_to_file`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `file_tasks` - One dict per file, the prescan tasks
/// * `universe_size` - Genes in the universe
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The multi-file dict, see [multi_dict].
#[pyfunction]
#[pyo3(signature = (f_path_cells, file_tasks, universe_size, qc_params, verbose = false))]
pub fn multi_h5ad_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    file_tasks: &Bound<'py, PyAny>,
    universe_size: usize,
    qc_params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let tasks: Vec<H5adFileTask> = from_py(file_tasks, "file_tasks")?;
    let res = py
        .detach(|| {
            h5ad_multifile_io::multi_h5ad_to_file(
                &tasks,
                &f_path_cells,
                universe_size,
                &qc,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    let per_file = res
        .per_file
        .iter()
        .map(|f| per_file_dict(py, &f.exp_id, Some(&f.cells_to_keep), &f.lib_size, &f.nnz))
        .collect::<PyResult<Vec<_>>>()?;
    multi_dict(
        py,
        &res.global_gene_indices,
        res.total_cells,
        res.total_genes,
        per_file,
    )
}

/// Load several mtx files into one cell-based binary
///
/// The twin of `multi_mtx_to_file`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `file_tasks` - One dict per file with `exp_id`, `mtx_path`,
///   `cells_as_rows` and `gene_local_to_universe`
/// * `universe_size` - Genes in the universe
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The multi-file dict, see [multi_dict].
#[pyfunction]
#[pyo3(signature = (f_path_cells, file_tasks, universe_size, qc_params, verbose = false))]
pub fn multi_mtx_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    file_tasks: &Bound<'py, PyAny>,
    universe_size: usize,
    qc_params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let tasks: Vec<MtxFileTask> = from_py(file_tasks, "file_tasks")?;
    let res = py
        .detach(|| {
            mtx_multifile_io::multi_mtx_to_file(&tasks, &f_path_cells, universe_size, &qc, verbose)
        })
        .map_err(BvErr::from)?;
    let per_file = res
        .per_file
        .iter()
        .map(|f| per_file_dict(py, &f.exp_id, Some(&f.cells_to_keep), &f.lib_size, &f.nnz))
        .collect::<PyResult<Vec<_>>>()?;
    multi_dict(
        py,
        &res.global_gene_indices,
        res.total_cells,
        res.total_genes,
        per_file,
    )
}

/// Load several 10x h5 files into one cell-based binary
///
/// The twin of `multi_tenx_h5_to_file`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `file_tasks` - One dict per file, the prescan tasks
/// * `universe_size` - Genes in the universe
/// * `qc_params` - `to_rust(MinQualityParams)`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The multi-file dict, see [multi_dict].
#[pyfunction]
#[pyo3(signature = (f_path_cells, file_tasks, universe_size, qc_params, verbose = false))]
pub fn multi_tenx_h5_to_file<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    file_tasks: &Bound<'py, PyAny>,
    universe_size: usize,
    qc_params: &Bound<'py, PyAny>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let qc: MinCellQuality = from_py(qc_params, "qc_params")?;
    let tasks: Vec<TenxFileTask> = from_py(file_tasks, "file_tasks")?;
    let res = py
        .detach(|| multi_10x_h5_to_file(&tasks, &f_path_cells, universe_size, &qc, verbose))
        .map_err(BvErr::from)?;
    let per_file = res
        .per_file
        .iter()
        .map(|f| per_file_dict(py, &f.exp_id, Some(&f.cells_to_keep), &f.lib_size, &f.nnz))
        .collect::<PyResult<Vec<_>>>()?;
    multi_dict(
        py,
        &res.global_gene_indices,
        res.total_cells,
        res.total_genes,
        per_file,
    )
}

/// Merge the cell-based binaries of several experiments into one
///
/// The twin of `merge_sc_files`.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to write
/// * `merge_tasks` - One dict per input with `exp_id`, `bin_cells_path`,
///   `cells_to_keep` and `gene_local_to_universe`
/// * `universe_size` - Genes in the intersection
/// * `renormalise` - Recompute the normalised layer against `target_size`
/// * `target_size` - Target library size, used with `renormalise`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `total_cells`, `total_genes` and `per_file` (`exp_id`, `lib_size`,
/// `nnz`).
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, merge_tasks, universe_size, renormalise, target_size, verbose = false
))]
pub fn merge_sc_files<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    merge_tasks: &Bound<'py, PyAny>,
    universe_size: usize,
    renormalise: bool,
    target_size: f32,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let tasks: Vec<BinMergeTask> = from_py(merge_tasks, "merge_tasks")?;
    let res = py
        .detach(|| {
            merge_sc_bin_files(
                &tasks,
                &f_path_cells,
                universe_size,
                renormalise,
                target_size,
                verbose,
            )
        })
        .map_err(BvErr::from)?;
    let per_file = res
        .per_file
        .iter()
        .map(|f| per_file_dict(py, &f.exp_id, None, &f.lib_size, &f.nnz))
        .collect::<PyResult<Vec<_>>>()?;
    let d = PyDict::new(py);
    d.set_item("total_cells", res.total_cells)?;
    d.set_item("total_genes", res.total_genes)?;
    d.set_item("per_file", PyList::new(py, per_file)?)?;
    Ok(d)
}

////////////////////////
// Store maintenance //
////////////////////////

/// Build the gene-based binary from the cell-based one
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary to read
/// * `f_path_genes` - Gene-based binary to write
/// * `max_mem_gb` - Memory for the conversion buffers in GB, `None` for one
///   phase
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `None`; raises on failure.
#[pyfunction]
#[pyo3(signature = (f_path_cells, f_path_genes, max_mem_gb = None, verbose = false))]
pub fn generate_gene_based_data(
    py: Python<'_>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    max_mem_gb: Option<f64>,
    verbose: bool,
) -> PyResult<()> {
    let cells = path_str(&f_path_cells)?;
    let genes = path_str(&f_path_genes)?;
    py.detach(|| write_gene_file(cells, genes, max_nnz(max_mem_gb), verbose))
        .map_err(BvErr::from)?;
    Ok(())
}

/// Number of cells expressing each gene
///
/// ### Params
///
/// * `f_path_genes` - Gene-based binary
///
/// ### Returns
///
/// `uint64` non-zero count per gene, in store order.
#[pyfunction]
pub fn gene_nnz<'py>(py: Python<'py>, f_path_genes: PathBuf) -> PyResult<Bound<'py, PyAny>> {
    let path = path_str(&f_path_genes)?;
    let nnz = py
        .detach(|| ParallelSparseReader::new(path)?.get_all_gene_nnz())
        .map_err(BvErr::from)?;
    Ok(u64_array(py, &nnz))
}

/// Archive the cell-based binary for cold storage
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary
/// * `f_path_archive` - Archive to write
/// * `level` - zstd level, 1 to 22
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `n_cells`, `nnz`, `n_norm_stored` and `archive_bytes`.
#[pyfunction]
#[pyo3(signature = (f_path_cells, f_path_archive, level = 3, verbose = false))]
pub fn archive_cells<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    f_path_archive: PathBuf,
    level: i32,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cells = path_str(&f_path_cells)?;
    let archive = path_str(&f_path_archive)?;
    let stats = py
        .detach(|| archive_cell_file(cells, archive, level, verbose))
        .map_err(BvErr::from)?;
    let d = PyDict::new(py);
    d.set_item("n_cells", stats.n_cells)?;
    d.set_item("nnz", stats.nnz)?;
    d.set_item("n_norm_stored", stats.n_norm_stored)?;
    d.set_item("archive_bytes", stats.archive_bytes)?;
    Ok(d)
}

/// Restore both binaries from an archive
///
/// ### Params
///
/// * `f_path_archive` - The archive
/// * `f_path_cells` - Cell-based binary to write
/// * `f_path_genes` - Gene-based binary to write
/// * `max_mem_gb` - Memory for the gene file conversion in GB
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `None`; raises on failure.
#[pyfunction]
#[pyo3(signature = (f_path_archive, f_path_cells, f_path_genes, max_mem_gb = None, verbose = false))]
pub fn restore_cells_archive(
    py: Python<'_>,
    f_path_archive: PathBuf,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    max_mem_gb: Option<f64>,
    verbose: bool,
) -> PyResult<()> {
    let archive = path_str(&f_path_archive)?;
    let cells = path_str(&f_path_cells)?;
    let genes = path_str(&f_path_genes)?;
    py.detach(|| restore_archive(archive, cells, genes, max_nnz(max_mem_gb), verbose))
        .map_err(BvErr::from)?;
    Ok(())
}

/// Read the file format version of a binary from its header
///
/// ### Params
///
/// * `path` - A cell- or gene-based binary
///
/// ### Returns
///
/// The version, e.g. 3 or 4.
#[pyfunction]
pub fn file_version(path: PathBuf) -> PyResult<u32> {
    Ok(peek_file_version(&path).map_err(BvErr::from)?)
}

/// Convert both binaries to the v4 format in place
///
/// ### Params
///
/// * `f_path_cells` - Cell-based binary
/// * `f_path_genes` - Gene-based binary
/// * `codec` - `"lz4"`, `"zstd"` or `"none"`
/// * `keep_backup` - Keep the originals as `<path>.v3.bak`
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `None`; raises on failure, leaving the store untouched.
#[pyfunction]
#[pyo3(signature = (f_path_cells, f_path_genes, codec = "lz4", keep_backup = true, verbose = false))]
pub fn upgrade_to_v4(
    py: Python<'_>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    codec: &str,
    keep_backup: bool,
    verbose: bool,
) -> PyResult<()> {
    let codec = parse_chunk_codec(codec).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown codec '{codec}'. Use one of 'lz4', 'zstd' or 'none'."
        ))
    })?;
    let cells = path_str(&f_path_cells)?;
    let genes = path_str(&f_path_genes)?;
    py.detach(|| upgrade_store_to_v4(cells, Some(genes), codec, keep_backup, verbose))
        .map_err(BvErr::from)?;
    Ok(())
}

/////////////////////
// obs / var to DB //
/////////////////////

/// Write the obs or var dataframe of an h5ad file to Parquet
///
/// The twin of `rs_h5ad_frame_to_parquet`.
///
/// ### Params
///
/// * `h5_path` - The h5ad file
/// * `frame` - `"obs"` or `"var"`
/// * `keep` - Optional 0-based rows to keep, in output order
/// * `id_col` - Column to use as identifier when the group has no index
/// * `out_path` - Parquet file to write
///
/// ### Returns
///
/// The table description, see [info_dict].
#[pyfunction]
#[pyo3(signature = (h5_path, frame, keep, id_col, out_path))]
pub fn h5ad_obs_var_to_parquet<'py>(
    py: Python<'py>,
    h5_path: PathBuf,
    frame: &str,
    keep: Option<PyReadonlyArray1<'py, u32>>,
    id_col: Option<String>,
    out_path: PathBuf,
) -> PyResult<Bound<'py, PyDict>> {
    let kind = frame_kind(frame)?;
    let keep = keep_rows(keep)?;
    let info = py
        .detach(|| {
            h5ad_frame_to_parquet(
                &h5_path,
                kind,
                keep.as_deref(),
                id_col.as_deref(),
                &out_path,
            )
        })
        .map_err(BvErr::from)?;
    info_dict(py, info)
}

/// Write the barcodes or features of a 10x h5 file to Parquet
///
/// The twin of `rs_tenx_frame_to_parquet`.
///
/// ### Params
///
/// * `h5_path` - The 10x h5 file
/// * `version` - `"v2"` or `"v3"`
/// * `frame` - `"obs"` or `"var"`
/// * `keep` - Optional 0-based rows to keep, in output order
/// * `out_path` - Parquet file to write
///
/// ### Returns
///
/// The table description, see [info_dict].
#[pyfunction]
#[pyo3(signature = (h5_path, version, frame, keep, out_path))]
pub fn tenx_obs_var_to_parquet<'py>(
    py: Python<'py>,
    h5_path: PathBuf,
    version: &str,
    frame: &str,
    keep: Option<PyReadonlyArray1<'py, u32>>,
    out_path: PathBuf,
) -> PyResult<Bound<'py, PyDict>> {
    let kind = frame_kind(frame)?;
    let version = tenx_version(version)?;
    let keep = keep_rows(keep)?;
    let info = py
        .detach(|| tenx_frame_to_parquet(&h5_path, version, kind, keep.as_deref(), &out_path))
        .map_err(BvErr::from)?;
    info_dict(py, info)
}

/// Snake case column names exactly as R's `to_snake_case()` does
///
/// ### Params
///
/// * `names` - The names
///
/// ### Returns
///
/// The snake_cased names, in order.
#[pyfunction]
pub fn snake_case(names: Vec<String>) -> Vec<String> {
    names.iter().map(|n| to_snake_case(n)).collect()
}

////////////
// Scans //
////////////

/// Dimensions and storage layout of an h5ad file
///
/// ### Params
///
/// * `h5_path` - The h5ad file
///
/// ### Returns
///
/// `(n_obs, n_var, format)` with `format` one of `"CSR"`, `"CSC"`,
/// `"DENSE_ROW"` or `"DENSE_COL"`.
#[pyfunction]
pub fn h5ad_dimensions(py: Python<'_>, h5_path: PathBuf) -> PyResult<(usize, usize, &'static str)> {
    let dims = py
        .detach(|| get_h5ad_dimensions(&h5_path))
        .map_err(BvErr::from)?;
    Ok((dims.n_obs, dims.n_var, dims.format.as_str()))
}

/// Detect which slot of an h5ad file holds raw counts
///
/// ### Params
///
/// * `h5_path` - The h5ad file
/// * `candidates` - Slot names to test in priority order
/// * `n_sample` - Values to sample per slot
/// * `threshold` - Minimum fraction of whole numbers among the non-zeros
///
/// ### Returns
///
/// The first qualifying slot name, or `None`.
#[pyfunction]
pub fn raw_count_slot(
    py: Python<'_>,
    h5_path: PathBuf,
    candidates: Vec<String>,
    n_sample: usize,
    threshold: f64,
) -> PyResult<Option<&'static str>> {
    let slots = candidates
        .iter()
        .map(|c| raw_slot(c))
        .collect::<PyResult<Vec<_>>>()?;
    let found = py
        .detach(|| detect_raw_count_slot(&h5_path, &slots, n_sample, threshold))
        .map_err(BvErr::from)?;
    Ok(found.map(slot_name))
}

/// Read a 1-D string dataset of an HDF5 file
///
/// ### Params
///
/// * `h5_path` - The file
/// * `dataset` - Path of the dataset inside it
///
/// ### Returns
///
/// The strings.
#[pyfunction]
pub fn h5_strings(py: Python<'_>, h5_path: PathBuf, dataset: &str) -> PyResult<Vec<String>> {
    Ok(py
        .detach(|| read_h5_strings(&h5_path, dataset))
        .map_err(BvErr::from)?)
}

/// Layout version and dimensions of a 10x h5 file
///
/// ### Params
///
/// * `h5_path` - The 10x h5 file
///
/// ### Returns
///
/// `(version, n_cells, n_genes)`, `n_genes` counting every modality.
#[pyfunction]
pub fn tenx_h5_metadata(
    py: Python<'_>,
    h5_path: PathBuf,
) -> PyResult<(&'static str, usize, usize)> {
    let (version, (n_cells, n_genes)) = py
        .detach(|| {
            let v = detect_tenx_version(&h5_path)?;
            Ok::<_, bixverse_sc_data::errors::ScDataErrors>((
                v,
                get_tenx_dimensions(&h5_path, Some(v))?,
            ))
        })
        .map_err(BvErr::from)?;
    Ok((tenx_name(version), n_cells, n_genes))
}

/// Write a cells x features count matrix as a 10x h5 file
///
/// ### Params
///
/// * `h5_path` - File to create
/// * `version` - `"v2"` or `"v3"`
/// * `data` - Integral counts, one per non-zero
/// * `indices` - 0-based feature index per non-zero
/// * `indptr` - Row pointers over cells
/// * `shape` - `(n_cells, n_features)`
/// * `barcodes` - One per cell
/// * `ids` - Feature ids
/// * `names` - Feature names
/// * `feature_types` - Feature types (v3); `None` for all gene expression
///
/// ### Returns
///
/// `None`; raises on inconsistent input.
#[pyfunction]
#[pyo3(signature = (
    h5_path, version, data, indices, indptr, shape, barcodes, ids, names, feature_types = None
))]
#[allow(clippy::too_many_arguments)]
pub fn write_tenx(
    py: Python<'_>,
    h5_path: PathBuf,
    version: &str,
    data: PyReadonlyArray1<'_, u32>,
    indices: PyReadonlyArray1<'_, u32>,
    indptr: PyReadonlyArray1<'_, u32>,
    shape: (usize, usize),
    barcodes: Vec<String>,
    ids: Vec<String>,
    names: Vec<String>,
    feature_types: Option<Vec<String>>,
) -> PyResult<()> {
    let version = tenx_version(version)?;
    if indptr.len() != shape.0 + 1 {
        return Err(PyValueError::new_err(
            "indptr must have n_cells + 1 entries",
        ));
    }
    let counts = CompressedSparseData2::<u32>::new_csr(
        data.as_slice()?,
        indices.as_slice()?,
        indptr.as_slice()?,
        None,
        shape,
    );
    py.detach(|| {
        let features = TenxFeatures {
            ids: &ids,
            names: &names,
            feature_types: feature_types.as_deref(),
        };
        write_tenx_h5(&h5_path, version, &counts, &barcodes, &features)
    })
    .map_err(BvErr::from)?;
    Ok(())
}
