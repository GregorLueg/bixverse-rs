//! Single cell ATAC: the bindings of `bixverse-atac`.
//!
//! Stores are written by path, regions travel as parallel `chrom`, `start`,
//! `end` lists (0-based, half-open), cells and features as 0-based `uint32`
//! store indices.

use std::path::PathBuf;

use bixverse_atac::annotation::{
    GeneActivityParams, gene_activity_regions, read_gtf_genes, tss_positions,
};
use bixverse_atac::counting::{count_fragments, parse_count_mode, write_counts_store};
use bixverse_atac::errors::AtacErrors;
use bixverse_atac::features::feature_total_stats;
use bixverse_atac::lsi::{LsiParams, lsi_on_store, set_tf_idf_on_store};
use bixverse_atac::qc::{FragmentQcParams, fragment_qc};
use bixverse_atac::regions::{genome_tiles, read_bed, reduce_regions, region};
use bixverse_atac::tenx_peaks::tenx_peaks_to_store;
use bixverse_sc::sc_processing::pca::parse_pca_solver;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_data::gene_file_io::GENE_FILE_BYTES_PER_NNZ;
use bixverse_sc_data::interval_index::FeatureInterval;
use bixverse_sc_data::v5_io::parse_feature_kind;
use numpy::{IntoPyArray, PyReadonlyArray1};
use py_common::convert::{f64_array, index_vec, mat_array, u32_array};
use py_common::error::BvErr;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/////////////
// Helpers //
/////////////

/// An `AtacErrors` as a Python exception.
///
/// ### Params
///
/// * `e` - The error
///
/// ### Returns
///
/// The store errors as their usual exceptions, I/O as `OSError`, the rest as
/// `ValueError`.
fn atac_err(e: AtacErrors) -> PyErr {
    match e {
        AtacErrors::Sc(e) => BvErr(e).into(),
        AtacErrors::ScData(e) => BvErr::from(e).into(),
        AtacErrors::Io(e) => PyIOError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// A path as `&str`.
///
/// ### Params
///
/// * `p` - The path
///
/// ### Returns
///
/// The string, or `ValueError` for invalid UTF-8.
fn path_str(p: &std::path::Path) -> PyResult<&str> {
    p.to_str()
        .ok_or_else(|| PyValueError::new_err(format!("{} is not valid UTF-8", p.display())))
}

/// Regions from parallel lists.
///
/// ### Params
///
/// * `chrom` - Chromosomes
/// * `start` - 0-based starts
/// * `end` - Exclusive ends
///
/// ### Returns
///
/// The regions, or `ValueError` for lists of different lengths or an empty
/// region.
fn regions_from(chrom: &[String], start: &[u32], end: &[u32]) -> PyResult<Vec<FeatureInterval>> {
    if chrom.len() != start.len() || chrom.len() != end.len() {
        return Err(PyValueError::new_err(
            "chrom, start and end must have the same length",
        ));
    }
    chrom
        .iter()
        .zip(start)
        .zip(end)
        .map(|((c, &s), &e)| {
            if e <= s {
                Err(PyValueError::new_err(format!("empty region {c}:{s}-{e}")))
            } else {
                Ok(region(c, s, e))
            }
        })
        .collect()
}

/// Regions as a dict of `chrom`, `start`, `end`.
///
/// ### Params
///
/// * `py` - GIL token
/// * `regions` - The regions
///
/// ### Returns
///
/// The dict.
fn regions_dict<'py>(py: Python<'py>, regions: &[FeatureInterval]) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "chrom",
        regions.iter().map(|r| r.chrom.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "start",
        regions
            .iter()
            .map(|r| r.start)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "end",
        regions
            .iter()
            .map(|r| r.end)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    Ok(out)
}

/// Memory budget of the CSR to CSC conversion as a non-zero count.
///
/// ### Params
///
/// * `max_mem_gb` - Budget in GB, `None` for one phase
///
/// ### Returns
///
/// Non-zeros per phase.
fn max_nnz(max_mem_gb: Option<f64>) -> Option<usize> {
    max_mem_gb.map(|gb| ((gb * 1e9) as usize / GENE_FILE_BYTES_PER_NNZ).max(1))
}

/////////////
// Regions //
/////////////

/// Read a BED file (first three columns).
///
/// ### Params
///
/// * `path` - The BED file, plain or gzip
///
/// ### Returns
///
/// Dict of `chrom`, `start`, `end`.
#[pyfunction]
pub fn atac_read_bed<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let regions = py.detach(|| read_bed(&path)).map_err(atac_err)?;
    regions_dict(py, &regions)
}

/// Fixed-width tiles over chromosomes.
///
/// ### Params
///
/// * `chroms` - Chromosome names
/// * `sizes` - Their lengths
/// * `width` - Tile width
///
/// ### Returns
///
/// Dict of `chrom`, `start`, `end`.
#[pyfunction]
pub fn atac_genome_tiles<'py>(
    py: Python<'py>,
    chroms: Vec<String>,
    sizes: Vec<u32>,
    width: u32,
) -> PyResult<Bound<'py, PyDict>> {
    if chroms.len() != sizes.len() {
        return Err(PyValueError::new_err("chroms and sizes differ in length"));
    }
    let cs: Vec<(String, u32)> = chroms.into_iter().zip(sizes).collect();
    regions_dict(py, &genome_tiles(&cs, width))
}

/// Merge overlapping and abutting regions.
///
/// ### Params
///
/// * `chrom` - Chromosomes
/// * `start` - Starts
/// * `end` - Ends
///
/// ### Returns
///
/// Dict of `chrom`, `start`, `end` of the merged regions and `members`, the
/// input indices each absorbed.
#[pyfunction]
pub fn atac_reduce_regions<'py>(
    py: Python<'py>,
    chrom: Vec<String>,
    start: Vec<u32>,
    end: Vec<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let regions = regions_from(&chrom, &start, &end)?;
    let (merged, members) = reduce_regions(&regions);
    let out = regions_dict(py, &merged)?;
    out.set_item("members", members)?;
    Ok(out)
}

////////////////
// Annotation //
////////////////

/// The genes of a GTF.
///
/// ### Params
///
/// * `path` - The GTF, plain or gzip
///
/// ### Returns
///
/// Dict of `gene_id`, `gene_name`, `biotype`, `chrom`, `start`, `end`
/// (0-based, half-open) and `strand` (`1`, `-1`, `0`).
#[pyfunction]
pub fn atac_gtf_genes<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let genes = py.detach(|| read_gtf_genes(&path)).map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "gene_id",
        genes.iter().map(|g| g.gene_id.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "gene_name",
        genes
            .iter()
            .map(|g| g.gene_name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "biotype",
        genes.iter().map(|g| g.biotype.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "chrom",
        genes.iter().map(|g| g.chrom.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "start",
        genes
            .iter()
            .map(|g| g.start)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "end",
        genes
            .iter()
            .map(|g| g.end)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "strand",
        genes
            .iter()
            .map(|g| g.strand)
            .collect::<Vec<i8>>()
            .into_pyarray(py),
    )?;
    Ok(out)
}

/// TSS positions from a GTF.
///
/// ### Params
///
/// * `path` - The GTF
/// * `biotypes` - Biotypes to keep, empty for all
///
/// ### Returns
///
/// Dict of `chrom` and `pos` (0-based).
#[pyfunction]
pub fn atac_tss_positions<'py>(
    py: Python<'py>,
    path: PathBuf,
    biotypes: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let genes = py.detach(|| read_gtf_genes(&path)).map_err(atac_err)?;
    let tss = tss_positions(&genes, &biotypes);
    let out = PyDict::new(py);
    out.set_item("chrom", tss.iter().map(|t| t.0.clone()).collect::<Vec<_>>())?;
    out.set_item(
        "pos",
        tss.iter()
            .map(|t| t.1)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    Ok(out)
}

/// Gene body plus promoter regions from a GTF (Signac `GeneActivity`).
///
/// ### Params
///
/// * `path` - The GTF
/// * `upstream` - Bases added upstream
/// * `downstream` - Bases added downstream
/// * `max_width` - Genes this long or longer are dropped
/// * `biotypes` - Biotypes to keep, empty for all
///
/// ### Returns
///
/// Dict of `name`, `chrom`, `start`, `end`.
#[pyfunction]
pub fn atac_gene_activity_regions<'py>(
    py: Python<'py>,
    path: PathBuf,
    upstream: u32,
    downstream: u32,
    max_width: u32,
    biotypes: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let genes = py.detach(|| read_gtf_genes(&path)).map_err(atac_err)?;
    let params = GeneActivityParams {
        upstream,
        downstream,
        max_width,
        biotypes,
    };
    let (regions, names) = gene_activity_regions(&genes, &params);
    let out = regions_dict(py, &regions)?;
    out.set_item("name", names)?;
    Ok(out)
}

////////////
// Ingest //
////////////

/// The `Peaks` modality of a 10x multiome h5 into a peak store.
///
/// ### Params
///
/// * `h5_path` - The Cell Ranger ARC h5
/// * `f_path_cells` - Cell-based file to write
/// * `f_path_genes` - Gene-based file to write
/// * `barcodes` - Cells in store order
/// * `max_mem_gb` - Budget of the gene file conversion
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict of `name`, `chrom`, `start`, `end` per peak and `lib_size`, `nnz`
/// per cell.
#[pyfunction]
#[pyo3(signature = (h5_path, f_path_cells, f_path_genes, barcodes, max_mem_gb = None, verbose = 1))]
pub fn atac_tenx_peaks_to_store<'py>(
    py: Python<'py>,
    h5_path: PathBuf,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    barcodes: Vec<String>,
    max_mem_gb: Option<f64>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (cells, genes) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let res = py
        .detach(|| {
            tenx_peaks_to_store(
                &h5_path,
                cells,
                genes,
                &barcodes,
                max_nnz(max_mem_gb),
                verbose > 0,
            )
        })
        .map_err(atac_err)?;
    let out = regions_dict(py, &res.intervals)?;
    out.set_item("name", res.peak_names)?;
    out.set_item("lib_size", u32_array(py, &res.lib_size))?;
    out.set_item("nnz", u32_array(py, &res.nnz))?;
    Ok(out)
}

/// Count fragments into features and write the store.
///
/// ### Params
///
/// * `fragments` - The fragment file
/// * `f_path_cells` - Cell-based file to write
/// * `f_path_genes` - Gene-based file to write
/// * `barcodes` - Cells in store order
/// * `chrom` - Feature chromosomes
/// * `start` - Feature starts
/// * `end` - Feature ends
/// * `mode` - `"insertion"`, `"paired-insertion"` or `"fragment"`
/// * `kind` - `"peak"`, `"tile"` or `"gene"`
/// * `target_size` - Target size of the initial log-CPM norm
/// * `max_mem_gb` - Budget of the gene file conversion
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Dict of `lib_size` and `nnz` per cell.
#[pyfunction]
#[pyo3(signature = (
    fragments, f_path_cells, f_path_genes, barcodes, chrom, start, end, mode, kind,
    target_size, max_mem_gb = None, verbose = 1
))]
#[allow(clippy::too_many_arguments)]
pub fn atac_fragments_to_store<'py>(
    py: Python<'py>,
    fragments: PathBuf,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    barcodes: Vec<String>,
    chrom: Vec<String>,
    start: Vec<u32>,
    end: Vec<u32>,
    mode: &str,
    kind: &str,
    target_size: f32,
    max_mem_gb: Option<f64>,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = parse_count_mode(mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "mode must be 'insertion', 'paired-insertion' or 'fragment', got '{mode}'"
        ))
    })?;
    let kind = parse_feature_kind(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown feature kind '{kind}'")))?;
    let features = regions_from(&chrom, &start, &end)?;
    let (cells, genes) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let (lib, nnz) = py
        .detach(|| {
            let counts = count_fragments(&fragments, &barcodes, &features, mode)?;
            let nnz: Vec<usize> = counts.indices.iter().map(|v| v.len()).collect();
            let lib = write_counts_store(
                &counts,
                &features,
                &barcodes,
                kind,
                target_size,
                cells,
                genes,
                max_nnz(max_mem_gb),
                verbose > 0,
            )?;
            Ok::<_, AtacErrors>((lib, nnz))
        })
        .map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item("lib_size", lib.into_pyarray(py))?;
    out.set_item("nnz", u32_array(py, &nnz))?;
    Ok(out)
}

////////
// QC //
////////

/// Per-cell QC from a fragment file.
///
/// ### Params
///
/// * `fragments` - The fragment file
/// * `barcodes` - Cells, output order
/// * `tss_chrom` - TSS chromosomes
/// * `tss_pos` - TSS positions (0-based)
/// * `mito_chroms` - Names of the mitochondrial genome
///
/// ### Returns
///
/// Dict of per-cell `n_fragments`, `n_reads`, `nucleosome_free`,
/// `mononucleosomal`, `n_mito` (`uint64`) and `nucleosome_signal`,
/// `tss_enrichment`, `mito_fraction` (`float64`).
#[pyfunction]
pub fn atac_fragment_qc<'py>(
    py: Python<'py>,
    fragments: PathBuf,
    barcodes: Vec<String>,
    tss_chrom: Vec<String>,
    tss_pos: Vec<u32>,
    mito_chroms: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    if tss_chrom.len() != tss_pos.len() {
        return Err(PyValueError::new_err(
            "tss_chrom and tss_pos differ in length",
        ));
    }
    let tss: Vec<(String, u32)> = tss_chrom.into_iter().zip(tss_pos).collect();
    let params = FragmentQcParams {
        mito_chroms,
        ..Default::default()
    };
    let qc = py
        .detach(|| fragment_qc(&fragments, &barcodes, &tss, &params))
        .map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item("nucleosome_signal", qc.nucleosome_signal().into_pyarray(py))?;
    out.set_item("tss_enrichment", qc.tss_enrichment().into_pyarray(py))?;
    out.set_item("mito_fraction", qc.mito_fraction().into_pyarray(py))?;
    out.set_item("n_fragments", qc.n_fragments.into_pyarray(py))?;
    out.set_item("n_reads", qc.n_reads.into_pyarray(py))?;
    out.set_item("nucleosome_free", qc.nucleosome_free.into_pyarray(py))?;
    out.set_item("mononucleosomal", qc.mononucleosomal.into_pyarray(py))?;
    out.set_item("n_mito", qc.n_mito.into_pyarray(py))?;
    Ok(out)
}

/////////////////////
// TF-IDF and LSI  //
/////////////////////

/// Switch a store to TF-IDF over a set of cells.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based file
/// * `f_path_genes` - Gene-based file
/// * `cells` - Cells for the IDF, `None` for all
/// * `scale` - Scale factor
///
/// ### Returns
///
/// The IDF as `float64`.
#[pyfunction]
#[pyo3(signature = (f_path_cells, f_path_genes, cells, scale))]
pub fn atac_set_tf_idf<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    cells: Option<PyReadonlyArray1<'py, u32>>,
    scale: f32,
) -> PyResult<Bound<'py, PyAny>> {
    let cells = cells.map(|c| index_vec(&c)).transpose()?;
    let (cf, gf) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let idf = py
        .detach(|| set_tf_idf_on_store(cf, gf, cells.as_deref(), scale))
        .map_err(atac_err)?;
    Ok(f64_array(py, &idf))
}

/// Total raw counts per feature and their empirical percentiles.
///
/// ### Params
///
/// * `f_path_genes` - Gene-based file
/// * `cells` - Cells to sum over, `None` for all
///
/// ### Returns
///
/// Dict of `count` and `percentile` (`float64`).
#[pyfunction]
#[pyo3(signature = (f_path_genes, cells))]
pub fn atac_feature_totals<'py>(
    py: Python<'py>,
    f_path_genes: PathBuf,
    cells: Option<PyReadonlyArray1<'py, u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let cells = cells.map(|c| index_vec(&c)).transpose()?;
    let gf = path_str(&f_path_genes)?;
    let totals = py
        .detach(|| {
            let reader = ParallelSparseReader::new(gf)?;
            feature_total_stats(&reader, cells.as_deref())
        })
        .map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item("count", totals.count.into_pyarray(py))?;
    out.set_item("percentile", totals.percentile.into_pyarray(py))?;
    Ok(out)
}

/// LSI of a TF-IDF store.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based file
/// * `f_path_genes` - Gene-based file
/// * `cells` - Cells, output row order
/// * `features` - Features to use
/// * `n_comp` - Components
/// * `solver` - `"auto"`, `"exact"`, `"randomised"` or `"covariance"`
/// * `scale_embeddings` - Z-score each component
/// * `scale_max` - Optional clip of the z-scores
/// * `seed` - Seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// Dict of `embedding` (`(cells, n_comp)` `float32`), `loadings`
/// (`(features, n_comp)` `float32`), `singular_values` and `depth_cor`
/// (`float64`).
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, f_path_genes, cells, features, n_comp, solver = "auto",
    scale_embeddings = true, scale_max = None, seed = 42, verbose = 0
))]
#[allow(clippy::too_many_arguments)]
pub fn atac_lsi<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    cells: PyReadonlyArray1<'py, u32>,
    features: PyReadonlyArray1<'py, u32>,
    n_comp: usize,
    solver: &str,
    scale_embeddings: bool,
    scale_max: Option<f32>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let solver = parse_pca_solver(solver).ok_or_else(|| {
        PyValueError::new_err(format!(
            "solver must be 'auto', 'exact', 'randomised' or 'covariance', got '{solver}'"
        ))
    })?;
    let cells = index_vec(&cells)?;
    let features = index_vec(&features)?;
    let (cf, gf) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let params = LsiParams::new(n_comp, solver, scale_embeddings, scale_max);
    let res = py
        .detach(|| lsi_on_store(cf, gf, &cells, &features, &params, seed, verbose))
        .map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item("embedding", mat_array(py, &res.embedding)?)?;
    out.set_item("loadings", mat_array(py, &res.loadings)?)?;
    out.set_item("singular_values", f64_array(py, &res.singular_values))?;
    out.set_item("depth_cor", f64_array(py, &res.depth_cor))?;
    Ok(out)
}
