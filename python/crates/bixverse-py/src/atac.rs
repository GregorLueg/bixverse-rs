//! Single cell ATAC: the bindings of `bixverse-atac`.
//!
//! Stores are written by path, regions travel as parallel `chrom`, `start`,
//! `end` lists (0-based, half-open), cells and features as 0-based `uint32`
//! store indices.

use std::path::PathBuf;

use bixverse_atac::errors::AtacErrors;
use bixverse_atac::integration::rlsi::{RlsiParams, rlsi_on_store};
use bixverse_atac::io::annotation::{
    GeneActivityParams, gene_activity_regions, read_gtf_genes, tss_positions,
};
use bixverse_atac::io::tenx_peaks::tenx_peaks_to_store;
use bixverse_atac::peaks::calling::{
    KeepDup, NarrowPeak, PeakCallParams, call_peaks, parse_tag_source,
};
use bixverse_atac::peaks::regions::{genome_tiles, read_bed, reduce_regions, region};
use bixverse_atac::processing::amulet::{AmuletParams, amulet};
use bixverse_atac::processing::counting::{count_fragments, parse_count_mode, write_counts_store};
use bixverse_atac::processing::features::feature_total_stats;
use bixverse_atac::processing::lsi::{
    LsiParams, lsi_on_store, project_lsi, read_tf_idf, set_tf_idf_on_store,
};
use bixverse_atac::processing::qc::{FragmentQcParams, fragment_qc};
use bixverse_sc::sc_processing::knn::KnnParams;
use bixverse_sc::sc_processing::pca::parse_pca_solver;
use bixverse_sc_data::data_io::ParallelSparseReader;
use bixverse_sc_data::gene_file_io::GENE_FILE_BYTES_PER_NNZ;
use bixverse_sc_data::interval_index::FeatureInterval;
use bixverse_sc_data::v5_io::parse_feature_kind;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use py_common::convert::{f64_array, from_py, index_vec, mat_array, mat_view, u32_array};
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
/// Dict of `name` (unique), `gene_id` (as in the GTF), `chrom`, `start`, `end`.
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
    let (regions, names, ids) = gene_activity_regions(&genes, &params);
    let out = regions_dict(py, &regions)?;
    out.set_item("name", names)?;
    out.set_item("gene_id", ids)?;
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

//////////////////
// Peak calling //
//////////////////

/// Call peaks per group of cells (MACS3 `callpeak`, no control).
///
/// ### Params
///
/// * `fragments` - The fragment file
/// * `barcodes` - Cells
/// * `groups` - Group of each cell, `0..n_groups`
/// * `tag_source` - `"insertion"` or `"fragment_start"`
/// * `keep_dup` - Tags kept per position, `None` for all
/// * `shift` - Tag shift before extension
/// * `extsize` - Tag extension
/// * `genome_size` - Effective genome size
/// * `llocal` - Local lambda window
/// * `q_value` - q-value cutoff
/// * `min_length` - Minimum peak length, `None` for `extsize`
/// * `max_gap` - Largest merged gap, `None` for MACS's tag size
/// * `max_mem_gb` - Tag budget of the first pass, `None` for one chromosome
///   at a time
/// * `tmp_dir` - Directory of the tag scratch file, `None` for the system one
///
/// ### Returns
///
/// Dict of `group` (`uint32`), `chrom`, `start`, `end`, `summit` (`uint32`,
/// 0-based) and `pileup`, `pscore`, `fold`, `qscore` (`float32`), groups in
/// order, peaks by chromosome name and position.
#[pyfunction]
#[pyo3(signature = (
    fragments, barcodes, groups, tag_source, keep_dup, shift, extsize,
    genome_size, llocal, q_value, min_length, max_gap, max_mem_gb, tmp_dir
))]
#[allow(clippy::too_many_arguments)]
pub fn atac_call_peaks<'py>(
    py: Python<'py>,
    fragments: PathBuf,
    barcodes: Vec<String>,
    groups: Vec<u32>,
    tag_source: &str,
    keep_dup: Option<u32>,
    shift: i32,
    extsize: u32,
    genome_size: f64,
    llocal: u32,
    q_value: f64,
    min_length: Option<u32>,
    max_gap: Option<u32>,
    max_mem_gb: Option<f64>,
    tmp_dir: Option<PathBuf>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_source = parse_tag_source(tag_source)
        .ok_or_else(|| PyValueError::new_err(format!("unknown tag_source '{tag_source}'")))?;
    let params = PeakCallParams {
        tag_source,
        keep_dup: keep_dup.map_or(KeepDup::All, KeepDup::AtMost),
        shift,
        extsize,
        genome_size,
        llocal,
        q_value,
        min_length,
        max_gap,
        max_mem_gb,
        tmp_dir,
    };
    let peaks = py
        .detach(|| call_peaks(&fragments, &barcodes, &groups, Some(params)))
        .map_err(atac_err)?;

    let flat: Vec<(u32, &NarrowPeak)> = peaks
        .iter()
        .enumerate()
        .flat_map(|(g, ps)| ps.iter().map(move |p| (g as u32, p)))
        .collect();
    let u32s = |f: fn(&NarrowPeak) -> u32| flat.iter().map(|(_, p)| f(p)).collect::<Vec<u32>>();
    let f32s = |f: fn(&NarrowPeak) -> f32| flat.iter().map(|(_, p)| f(p)).collect::<Vec<f32>>();
    let out = PyDict::new(py);
    out.set_item(
        "group",
        flat.iter()
            .map(|&(g, _)| g)
            .collect::<Vec<u32>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "chrom",
        flat.iter()
            .map(|(_, p)| p.chrom.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("start", u32s(|p| p.start).into_pyarray(py))?;
    out.set_item("end", u32s(|p| p.end).into_pyarray(py))?;
    out.set_item("summit", u32s(|p| p.summit).into_pyarray(py))?;
    out.set_item("pileup", f32s(|p| p.pileup).into_pyarray(py))?;
    out.set_item("pscore", f32s(|p| p.pscore).into_pyarray(py))?;
    out.set_item("fold", f32s(|p| p.fold).into_pyarray(py))?;
    out.set_item("qscore", f32s(|p| p.qscore).into_pyarray(py))?;
    Ok(out)
}

////////////////
// Multiplets //
////////////////

/// AMULET multiplet detection from fragment overlaps.
///
/// ### Params
///
/// * `fragments` - The fragment file
/// * `barcodes` - Cells to test
/// * `restrict_beds` - BED files of regions to drop (repeats, blacklists)
/// * `max_overlap` - Overlapping fragments a diploid cell can explain
/// * `max_fragment_len` - Longest valid fragment
/// * `repeat_q` - q-value under which a site is a repeat
/// * `multiplet_q` - q-value under which a cell is a multiplet
/// * `chroms` - Chromosomes to use, `None` for the autosomes
/// * `max_mem_gb` - Read budget, `None` for one chromosome at a time
/// * `count_trailing_loci` - Count loci no later fragment passes
///
/// ### Returns
///
/// Dict of per-cell `n_fragments`, `n_overlaps`, `n_sites` (`uint32`),
/// `p_value`, `q_value` (`float64`), `multiplet` (`bool`), and the totals
/// `n_merged_sites`, `n_repeat_sites`.
#[pyfunction]
#[pyo3(signature = (
    fragments, barcodes, restrict_beds, max_overlap, max_fragment_len, repeat_q,
    multiplet_q, chroms, max_mem_gb, count_trailing_loci
))]
#[allow(clippy::too_many_arguments)]
pub fn atac_amulet<'py>(
    py: Python<'py>,
    fragments: PathBuf,
    barcodes: Vec<String>,
    restrict_beds: Vec<PathBuf>,
    max_overlap: u32,
    max_fragment_len: u32,
    repeat_q: f64,
    multiplet_q: f64,
    chroms: Option<Vec<String>>,
    max_mem_gb: Option<f64>,
    count_trailing_loci: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params = AmuletParams {
        max_overlap,
        max_fragment_len,
        repeat_q,
        multiplet_q,
        chroms,
        max_mem_gb,
        count_trailing_loci,
    };
    let res = py
        .detach(|| {
            let mut restrict = Vec::new();
            for bed in &restrict_beds {
                restrict.extend(read_bed(bed)?);
            }
            amulet(&fragments, &barcodes, &restrict, Some(params))
        })
        .map_err(atac_err)?;
    let out = PyDict::new(py);
    out.set_item("n_fragments", res.n_fragments.into_pyarray(py))?;
    out.set_item("n_overlaps", res.n_overlaps.into_pyarray(py))?;
    out.set_item("n_sites", res.n_sites.into_pyarray(py))?;
    out.set_item("p_value", res.p_value.into_pyarray(py))?;
    out.set_item("q_value", res.q_value.into_pyarray(py))?;
    out.set_item("multiplet", res.multiplet.into_pyarray(py))?;
    out.set_item("n_merged_sites", res.n_merged_sites)?;
    out.set_item("n_repeat_sites", res.n_repeat_sites)?;
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

/// Project cells of a TF-IDF store onto a fitted LSI (Seurat
/// `ProjectSVD(mode = "lsi")`, no centring or scaling).
///
/// ### Params
///
/// * `f_path_cells` - Cell-based file of the query store
/// * `f_path_genes` - Gene-based file of the query store (TF-IDF norm)
/// * `cells` - Query cells, output row order
/// * `features` - Query features, in the row order of `loadings`
/// * `loadings` - `(features, n_comp)` loadings of the fit
/// * `singular_values` - Singular values of the fit
///
/// ### Returns
///
/// `(cells, n_comp)` `float32`.
#[pyfunction]
pub fn atac_project_lsi<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    cells: PyReadonlyArray1<'py, u32>,
    features: PyReadonlyArray1<'py, u32>,
    loadings: PyReadonlyArray2<'py, f32>,
    singular_values: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyAny>> {
    let cells = index_vec(&cells)?;
    let features = index_vec(&features)?;
    let loadings = mat_view(&loadings)?;
    let sv = singular_values.as_slice()?;
    let (cf, gf) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let res = py
        .detach(|| {
            let tf_idf = read_tf_idf(cf, gf, &cells, &features, None)?;
            project_lsi(&tf_idf, loadings, sv)
        })
        .map_err(atac_err)?;
    mat_array(py, &res)
}

/// rLSI integration of an embedding over a peak or tile store.
///
/// ### Params
///
/// * `f_path_cells` - Cell-based file of the store
/// * `f_path_genes` - Gene-based file of the store
/// * `cells` - Cells, rows of `embedding`
/// * `features` - Features of the per-batch LSIs
/// * `batch` - 0-based batch per cell
/// * `embedding` - `(cells, d)` embedding to integrate
/// * `dims` - 0-based LSI components the anchors use
/// * `n_comp` - Components of the per-batch LSIs
/// * `solver` - SVD solver
/// * `k_anchor` - Neighbours for the anchors
/// * `k_score` - Neighbours for the anchor scores
/// * `k_weight` - Anchors per cell in the correction weights
/// * `sd` - Bandwidth divisor of the weights
/// * `l2_norm` - Scale and L2 normalise the anchor spaces
/// * `knn_params` - `to_rust(KnnParams)`
/// * `seed` - Seed
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The integrated `(cells, d)` `float32` embedding.
#[pyfunction]
#[pyo3(signature = (
    f_path_cells, f_path_genes, cells, features, batch, embedding, dims, n_comp, solver,
    k_anchor, k_score, k_weight, sd, l2_norm, knn_params, seed = 42, verbose = 0
))]
#[allow(clippy::too_many_arguments)]
pub fn atac_rlsi<'py>(
    py: Python<'py>,
    f_path_cells: PathBuf,
    f_path_genes: PathBuf,
    cells: PyReadonlyArray1<'py, u32>,
    features: PyReadonlyArray1<'py, u32>,
    batch: PyReadonlyArray1<'py, u32>,
    embedding: PyReadonlyArray2<'py, f32>,
    dims: PyReadonlyArray1<'py, u32>,
    n_comp: usize,
    solver: &str,
    k_anchor: usize,
    k_score: usize,
    k_weight: usize,
    sd: f32,
    l2_norm: bool,
    knn_params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let solver = parse_pca_solver(solver).ok_or_else(|| {
        PyValueError::new_err(format!(
            "solver must be 'auto', 'exact', 'randomised' or 'covariance', got '{solver}'"
        ))
    })?;
    let params = RlsiParams {
        dims: index_vec(&dims)?,
        k_anchor,
        k_score,
        k_weight,
        sd,
        l2_norm,
        lsi: LsiParams::new(n_comp, solver, true, None),
        knn_params: from_py::<KnnParams>(knn_params, "knn params")?,
    };
    let cells = index_vec(&cells)?;
    let features = index_vec(&features)?;
    let batch = index_vec(&batch)?;
    let embedding = mat_view(&embedding)?.to_owned();
    let (cf, gf) = (path_str(&f_path_cells)?, path_str(&f_path_genes)?);
    let res = py
        .detach(|| {
            rlsi_on_store(
                cf, gf, &cells, &features, &batch, &embedding, &params, seed, verbose,
            )
        })
        .map_err(atac_err)?;
    mat_array(py, &res)
}
