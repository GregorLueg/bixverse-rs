//! Meta cells: the Python twins of `rs_get_metacells_bootstrapped`,
//! `rs_get_seacells`, `rs_supercell`, `rs_metacells2`, `rs_metacell_density`,
//! `rs_metacell_compactness`, `rs_metacell_separation`, `rs_mc_hvg`,
//! `rs_mc_pca`, `rs_mc_fit_residuals`, `rs_mc_residual_variance` and
//! `rs_mc_pca_residuals`.
//!
//! The generators and the in-memory residual stages live in `bixverse-sc`;
//! this only moves buffers. Every index is 0-based. A meta cell matrix
//! crosses as its CSR parts, `indptr` as `uint32`.

use bixverse_sc::mc_analysis::metacell_density::{
    DensityRegion, compute_compactness, compute_diffusion_density, compute_separation,
};
use bixverse_sc::mc_generation::generate::{
    MetaCellsRes, bootstrapped_metacells_on_store, metacells2_on_store, seacells_on_store,
    supercells_on_store,
};
use bixverse_sc::mc_generation::hdwgcna_meta_cells::BootstrappedMetaCellParams;
use bixverse_sc::mc_generation::metacells2::{DacParams, MetacellsParams};
use bixverse_sc::mc_generation::seacells::SEACellsParams;
use bixverse_sc::mc_generation::super_cells::SuperCellParams;
use bixverse_sc::mc_processing::hvg_pca::{hvg_on_metacells, pca_on_metacells};
use bixverse_sc::prelude::KnnParams;
use bixverse_sc::sc_processing::analytic_pearson::model::AprParams;
use bixverse_sc::sc_processing::hvg::HvgParams;
use bixverse_sc::sc_processing::pca::SingleCellPcaParams;
use bixverse_sc::sc_processing::residual_fit::{
    ResidualParams, fit_residuals_in_memory, pca_residuals_in_memory, residual_variance_in_memory,
};
use bixverse_sc::sc_processing::sctransform::model::{SctCovariates, SctParams};
use bixverse_sc::sc_processing::sctransform::stream::SctStreamOpts;
use bixverse_sc_data::prelude::{BixverseNumeric, CompressedSparseData2, CompressedSparseFormat};
use faer::MatRef;
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::BvErr;
use crate::graph::{from_py, mat_view};
use crate::integrate::{dist_rows, knn_rows};
use crate::pp::{PyResidualFit, index_vec, mat_array, stats_dict};

///////////
// Types //
///////////

/// A meta cell matrix as its CSR parts: `(data, indices, indptr, shape)`.
pub(crate) type CsrParts<'a, 'py> = (
    &'a PyReadonlyArray1<'py, f32>,
    &'a PyReadonlyArray1<'py, u32>,
    &'a PyReadonlyArray1<'py, u32>,
    (usize, usize),
);

/////////////
// Helpers //
/////////////

/// A CSR matrix from its parts, the values in both layers
///
/// The residual and HVG kernels pick their layer by the element type, so
/// both carry the one matrix handed over, as the R side does.
///
/// ### Params
///
/// * `parts` - The CSR parts
/// * `cast` - Converts a value to the first layer's type
///
/// ### Returns
///
/// The matrix, or a `ValueError` when the parts do not line up.
pub(crate) fn csr_from_parts<T: BixverseNumeric>(
    parts: CsrParts<'_, '_>,
    cast: impl Fn(f32) -> T,
) -> PyResult<CompressedSparseData2<T, f32>> {
    let (data, indices, indptr, shape) = parts;
    let (data, indices, indptr) = (data.as_slice()?, indices.as_slice()?, indptr.as_slice()?);
    if indptr.len() != shape.0 + 1
        || data.len() != indices.len()
        || indptr.last().copied().unwrap_or(0) as usize != data.len()
    {
        return Err(PyValueError::new_err(
            "data, indices, indptr and shape do not describe one CSR matrix",
        ));
    }
    Ok(CompressedSparseData2::from_parts(
        data.iter().map(|&x| cast(x)).collect(),
        indices.to_vec(),
        indptr.to_vec(),
        Some(data.to_vec()),
        CompressedSparseFormat::Csr,
        shape,
    ))
}

/// Optional kNN indices and distances as one row per cell
///
/// ### Params
///
/// * `indices` - `(n, k)` neighbour indices
/// * `distances` - `(n, k)` distances
///
/// ### Returns
///
/// Both as rows, `None` without indices.
#[allow(clippy::type_complexity)]
pub(crate) fn knn_pair(
    indices: Option<&PyReadonlyArray2<'_, u32>>,
    distances: Option<&PyReadonlyArray2<'_, f32>>,
) -> PyResult<Option<(Vec<Vec<usize>>, Vec<Vec<f32>>)>> {
    match (indices, distances) {
        (Some(idx), Some(dist)) => Ok(Some((knn_rows(idx)?, dist_rows(dist, idx.shape())?))),
        (None, None) => Ok(None),
        _ => Err(PyValueError::new_err(
            "pass the kNN indices and distances together",
        )),
    }
}

/// The meta cells and their aggregated counts as a dict
///
/// ### Params
///
/// * `py` - GIL token
/// * `res` - The generator result
///
/// ### Returns
///
/// Dict with `metacells` (one `uint32` array of store cells per meta cell),
/// the CSR parts `indptr`, `indices`, `raw` and `norm` (`float32`), `shape`
/// and `n_total`.
fn res_dict<'py>(py: Python<'py>, res: MetaCellsRes) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let metacells = PyList::empty(py);
    for mc in res.metacells {
        let mc: Vec<u32> = mc.into_iter().map(|x| x as u32).collect();
        metacells.append(mc.into_pyarray(py))?;
    }
    let agg = res.aggregated;
    let raw: Vec<f32> = agg.data.iter().map(|&x| x as f32).collect();
    d.set_item("metacells", metacells)?;
    d.set_item("indptr", agg.indptr.into_pyarray(py))?;
    d.set_item("indices", agg.indices.into_pyarray(py))?;
    d.set_item("raw", raw.into_pyarray(py))?;
    d.set_item(
        "norm",
        agg.data_2
            .expect("aggregation writes the normalised layer")
            .into_pyarray(py),
    )?;
    d.set_item("shape", agg.shape)?;
    d.set_item("n_total", res.n_total)?;
    Ok(d)
}

/// Seed and verbosity of the in-memory residual stages
///
/// ### Params
///
/// * `seed` - Seed for the step-1 subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The options, the gene batching left at its default.
fn mc_opts(seed: u64, verbose: bool) -> SctStreamOpts {
    SctStreamOpts {
        seed,
        verbose: verbose as usize,
        ..SctStreamOpts::default()
    }
}

////////////////
// Generators //
////////////////

/// Bootstrapped (hdWGCNA) meta cells
///
/// The twin of `rs_get_metacells_bootstrapped`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `knn` - Optional `(n, k)` kNN over the working rows
/// * `embd` - Optional `(n, d)` embedding, C-contiguous
/// * `cells_to_keep` - Store index of each working row
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - `to_rust(BtMetacellsParams)`
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [res_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cell, knn, embd, cells_to_keep, cells_to_use, params, target_size, seed,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrapped_metacells<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    knn: Option<PyReadonlyArray2<'py, u32>>,
    embd: Option<PyReadonlyArray2<'py, f32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    cells_to_use: Option<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    target_size: f32,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: BootstrappedMetaCellParams = from_py(params, "bootstrapped meta cell params")?;
    let knn = knn.as_ref().map(knn_rows).transpose()?;
    let embd: Option<MatRef<f32>> = embd.as_ref().map(mat_view).transpose()?;
    let keep = index_vec(&cells_to_keep)?;
    let to_use = cells_to_use.as_ref().map(index_vec).transpose()?;
    let res = py
        .detach(|| {
            bootstrapped_metacells_on_store(
                f_path_cell,
                knn.as_deref(),
                embd,
                Some(&keep),
                to_use.as_deref(),
                &params,
                target_size,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    res_dict(py, res)
}

/// SEACells
///
/// The twin of `rs_get_seacells`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `embd` - `(n, d)` embedding, C-contiguous
/// * `knn_indices` - Optional `(n, k)` kNN over the working rows
/// * `knn_distances` - Its distances
/// * `cells_to_keep` - Store index of each working row
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - `to_rust(SeacellsParams)`
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [res_dict], plus `rss` and `archetypes` (store indices).
#[pyfunction]
#[pyo3(signature = (
    f_path_cell, embd, knn_indices, knn_distances, cells_to_keep, cells_to_use, params,
    target_size, seed, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn seacells<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    embd: PyReadonlyArray2<'py, f32>,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_distances: Option<PyReadonlyArray2<'py, f32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    cells_to_use: Option<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    target_size: f32,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: SEACellsParams = from_py(params, "SEACells params")?;
    let knn = knn_pair(knn_indices.as_ref(), knn_distances.as_ref())?;
    let embd = mat_view(&embd)?;
    let keep = index_vec(&cells_to_keep)?;
    let to_use = cells_to_use.as_ref().map(index_vec).transpose()?;
    let res = py
        .detach(|| {
            seacells_on_store(
                f_path_cell,
                embd,
                knn.as_ref().map(|(i, d)| (i.as_slice(), d.as_slice())),
                Some(&keep),
                to_use.as_deref(),
                &params,
                target_size,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    let archetypes: Vec<u32> = res.archetypes.iter().map(|&x| x as u32).collect();
    let d = res_dict(py, res.base)?;
    d.set_item("rss", res.rss.into_pyarray(py))?;
    d.set_item("archetypes", archetypes.into_pyarray(py))?;
    Ok(d)
}

/// SuperCells
///
/// The twin of `rs_supercell`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `knn_indices` - Optional `(n, k)` kNN over the working rows
/// * `knn_distances` - Its distances
/// * `embd` - Optional `(n, d)` embedding, C-contiguous
/// * `cells_to_keep` - Store index of each working row
/// * `cells_to_use` - Optional narrowing, as store indices
/// * `params` - `to_rust(SupercellParams)`
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed for a kNN build
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [res_dict].
#[pyfunction]
#[pyo3(signature = (
    f_path_cell, knn_indices, knn_distances, embd, cells_to_keep, cells_to_use, params,
    target_size, seed, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn supercells<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    knn_indices: Option<PyReadonlyArray2<'py, u32>>,
    knn_distances: Option<PyReadonlyArray2<'py, f32>>,
    embd: Option<PyReadonlyArray2<'py, f32>>,
    cells_to_keep: PyReadonlyArray1<'py, u32>,
    cells_to_use: Option<PyReadonlyArray1<'py, u32>>,
    params: &Bound<'py, PyAny>,
    target_size: f32,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params: SuperCellParams = from_py(params, "SuperCell params")?;
    let knn = knn_pair(knn_indices.as_ref(), knn_distances.as_ref())?;
    let embd: Option<MatRef<f32>> = embd.as_ref().map(mat_view).transpose()?;
    let keep = index_vec(&cells_to_keep)?;
    let to_use = cells_to_use.as_ref().map(index_vec).transpose()?;
    let res = py
        .detach(|| {
            supercells_on_store(
                f_path_cell,
                knn.as_ref().map(|(i, d)| (i.as_slice(), d.as_slice())),
                embd,
                Some(&keep),
                to_use.as_deref(),
                &params,
                target_size,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    res_dict(py, res)
}

/// MetaCells2 (divide and conquer)
///
/// The twin of `rs_metacells2`.
///
/// ### Params
///
/// * `f_path_cell` - The cell-based binary
/// * `cell_indices` - Store indices of the cells to include
/// * `mc2_params` - `to_rust(Metacells2Params)`
/// * `dac_params` - `to_rust(Metacells2DacParams)`
/// * `target_size` - Library size to normalise the aggregates to
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// See [res_dict], plus per included cell `dissolved`, `level`,
/// `rare_module_of_cell` (0-based, `-1` for none), per store gene
/// `rare_module_of_gene`, and `selected_genes`.
#[pyfunction]
#[pyo3(signature = (
    f_path_cell, cell_indices, mc2_params, dac_params, target_size, seed, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn metacells2<'py>(
    py: Python<'py>,
    f_path_cell: &str,
    cell_indices: PyReadonlyArray1<'py, u32>,
    mc2_params: &Bound<'py, PyAny>,
    dac_params: &Bound<'py, PyAny>,
    target_size: f32,
    seed: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mc2: MetacellsParams = from_py(mc2_params, "MetaCells2 params")?;
    let dac: DacParams = from_py(dac_params, "MetaCells2 divide and conquer params")?;
    let cells = index_vec(&cell_indices)?;
    let res = py
        .detach(|| metacells2_on_store(f_path_cell, &cells, &mc2, &dac, target_size, seed, verbose))
        .map_err(BvErr::from)?;
    let selected: Vec<u32> = res.selected_genes.iter().map(|&g| g as u32).collect();
    let d = res_dict(py, res.base)?;
    d.set_item("dissolved", res.dissolved.into_pyarray(py))?;
    d.set_item("level", res.level.into_pyarray(py))?;
    d.set_item(
        "rare_module_of_cell",
        res.rare_module_of_cell.into_pyarray(py),
    )?;
    d.set_item(
        "rare_module_of_gene",
        res.rare_module_of_gene.into_pyarray(py),
    )?;
    d.set_item("selected_genes", selected.into_pyarray(py))?;
    Ok(d)
}

/////////////
// Metrics //
/////////////

/// Diffusion components and density regions over a kNN graph
///
/// The twin of `rs_metacell_density`.
///
/// ### Params
///
/// * `knn_indices` - `(n, k)` kNN
/// * `knn_distances` - Its distances
/// * `n_dcs` - Diffusion components to return
/// * `k_density` - The neighbour in diffusion space giving the density
/// * `knn_params` - `to_rust(KnnDefaults)` for the search in diffusion space
/// * `seed` - Seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(dcs, density_distances, regions)`: `(n, n_dcs)` `float32`, `float32`
/// per cell and `"high"`, `"mid"` or `"low"` per cell.
#[pyfunction]
#[pyo3(signature = (
    knn_indices, knn_distances, n_dcs, k_density, knn_params, seed, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn metacell_density<'py>(
    py: Python<'py>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_distances: PyReadonlyArray2<'py, f32>,
    n_dcs: usize,
    k_density: usize,
    knn_params: &Bound<'py, PyAny>,
    seed: u64,
    verbose: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Vec<&'static str>)> {
    let params: KnnParams = from_py(knn_params, "kNN params")?;
    let idx = knn_rows(&knn_indices)?;
    let dist = dist_rows(&knn_distances, knn_indices.shape())?;
    let res = py
        .detach(|| {
            compute_diffusion_density(
                &idx,
                &dist,
                n_dcs,
                k_density,
                &params,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    let regions = res
        .regions
        .iter()
        .map(|r| match r {
            DensityRegion::High => "high",
            DensityRegion::Mid => "mid",
            DensityRegion::Low => "low",
        })
        .collect();
    Ok((
        mat_array(py, &res.dcs)?,
        res.density_distances.into_pyarray(py).into_any(),
        regions,
    ))
}

/// Meta cell compactness in a diffusion map
///
/// The twin of `rs_metacell_compactness`.
///
/// ### Params
///
/// * `dcs` - `(n, n_dcs)` diffusion components, C-contiguous
/// * `metacells` - 0-based rows of `dcs` per meta cell
///
/// ### Returns
///
/// One value per meta cell, `NaN` for an empty one.
#[pyfunction]
pub fn metacell_compactness<'py>(
    py: Python<'py>,
    dcs: PyReadonlyArray2<'py, f32>,
    metacells: Vec<PyReadonlyArray1<'py, u32>>,
) -> PyResult<Bound<'py, PyAny>> {
    let groups = metacells
        .iter()
        .map(index_vec)
        .collect::<PyResult<Vec<_>>>()?;
    let dcs = mat_view(&dcs)?;
    Ok(compute_compactness(dcs, &groups)
        .into_pyarray(py)
        .into_any())
}

/// Meta cell separation in a diffusion map
///
/// The twin of `rs_metacell_separation`.
///
/// ### Params
///
/// * `dcs` - `(n, n_dcs)` diffusion components, C-contiguous
/// * `metacells` - 0-based rows of `dcs` per meta cell
///
/// ### Returns
///
/// One value per meta cell, `NaN` for an empty one and `inf` for a lone
/// non-empty one.
#[pyfunction]
pub fn metacell_separation<'py>(
    py: Python<'py>,
    dcs: PyReadonlyArray2<'py, f32>,
    metacells: Vec<PyReadonlyArray1<'py, u32>>,
) -> PyResult<Bound<'py, PyAny>> {
    let groups = metacells
        .iter()
        .map(index_vec)
        .collect::<PyResult<Vec<_>>>()?;
    let dcs = mat_view(&dcs)?;
    Ok(compute_separation(dcs, &groups).into_pyarray(py).into_any())
}

////////////////
// Processing //
////////////////

/// HVG statistics of a meta cell matrix
///
/// The twin of `rs_mc_hvg`.
///
/// ### Params
///
/// * `data` - CSR values: raw counts for `"vst"`, normalised otherwise
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `params` - `to_rust(HvgParams)`, method other than `"residual"`
///
/// ### Returns
///
/// Column name to one value per gene, as `hvg` returns them.
#[pyfunction]
pub fn mc_hvg<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let params: HvgParams = from_py(params, "hvg params")?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?;
    let res = py
        .detach(|| hvg_on_metacells(&m, &params))
        .map_err(BvErr::from)?;
    stats_dict(py, vec![res], false)
}

/// PCA on a meta cell matrix
///
/// The twin of `rs_mc_pca`.
///
/// ### Params
///
/// * `data` - CSR normalised values of the genes to use
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes_used)`
/// * `no_pcs` - Number of components
/// * `params` - `to_rust(PcaParams)`
/// * `clr_offsets` - Per meta cell offsets over the full gene panel; needed
///   with `clr`
/// * `seed` - Seed for the randomised SVD
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`, `float32`.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, no_pcs, params, clr_offsets, seed))]
#[allow(clippy::too_many_arguments)]
pub fn mc_pca<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    no_pcs: usize,
    params: &Bound<'py, PyAny>,
    clr_offsets: Option<PyReadonlyArray1<'py, f64>>,
    seed: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let params: SingleCellPcaParams = from_py(params, "pca params")?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x)?;
    let offsets = match (params.clr, clr_offsets) {
        (true, Some(o)) => Some(o.as_slice()?.to_vec()),
        (true, None) => return Err(PyValueError::new_err("clr needs clr_offsets")),
        (false, _) => None,
    };
    let (scores, loadings, s) = py
        .detach(|| pca_on_metacells(&m, no_pcs, &params, offsets.as_deref(), seed))
        .map_err(BvErr::from)?;
    Ok((
        mat_array(py, &scores)?,
        mat_array(py, &loadings)?,
        s.into_pyarray(py).into_any(),
    ))
}

/// Fit a residual model over a meta cell matrix
///
/// The twin of `rs_mc_fit_residuals`.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `method` - `"sctransform"` or `"analytic_pearson"`
/// * `group_of_cell` - Optional 0-based group per meta cell, dense
/// * `covariates` - `(name, values)` per covariate (scTransform only)
/// * `params` - `to_rust(SctransformParams)` or `to_rust(AprParams)`
/// * `seed` - Seed for the step-1 subsampling
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// The [PyResidualFit], its cells `0..n_metacells`.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, method, group_of_cell, covariates, params, seed = 42,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_fit_residuals(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, u32>,
    indptr: PyReadonlyArray1<'_, u32>,
    shape: (usize, usize),
    method: &str,
    group_of_cell: Option<PyReadonlyArray1<'_, u32>>,
    covariates: Vec<(String, PyReadonlyArray1<'_, f64>)>,
    params: &Bound<'_, PyAny>,
    seed: u64,
    verbose: bool,
) -> PyResult<PyResidualFit> {
    let params = match method {
        "sctransform" => ResidualParams::Sct(from_py::<SctParams>(params, "sctransform params")?),
        "analytic_pearson" => ResidualParams::Apr(from_py::<AprParams>(params, "apr params")?),
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'sctransform' or 'analytic_pearson', got '{other}'"
            )));
        }
    };
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let groups: Option<Vec<u32>> = group_of_cell
        .map(|g| g.as_slice().map(<[u32]>::to_vec))
        .transpose()?;
    let columns = covariates
        .iter()
        .map(|(name, v)| Ok((name.clone(), v.as_slice()?.to_vec())))
        .collect::<PyResult<Vec<(String, Vec<f64>)>>>()?;
    let fit = py
        .detach(|| {
            let covariates = SctCovariates::from_columns(&columns)?;
            fit_residuals_in_memory(
                &m,
                groups.as_deref(),
                covariates,
                params,
                mc_opts(seed, verbose),
            )
        })
        .map_err(BvErr::from)?;
    Ok(PyResidualFit { inner: fit })
}

/// Residual variance per gene and group, and the residual HVGs, of a meta
/// cell matrix
///
/// The twin of `rs_mc_residual_variance`.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `fit` - A fit from [mc_fit_residuals]
/// * `n_hvg` - Variable features per group
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(genes, variance, hvg)`, as `residual_variance`.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shape, fit, n_hvg, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn mc_residual_variance<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    fit: &PyResidualFit,
    n_hvg: usize,
    verbose: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let opts = mc_opts(SctStreamOpts::default().seed, verbose);
    let res = py
        .detach(|| residual_variance_in_memory(&m, &fit.inner, n_hvg, opts))
        .map_err(BvErr::from)?;
    let (n_genes, n_groups) = (res.genes.len(), res.variance.len());
    let flat: Vec<f64> = (0..n_genes)
        .flat_map(|g| res.variance.iter().map(move |v| v[g]))
        .collect();
    let genes: Vec<u32> = res.genes.iter().map(|&g| g as u32).collect();
    let hvg: Vec<u32> = res.hvg.iter().map(|&g| g as u32).collect();
    Ok((
        genes.into_pyarray(py).into_any(),
        flat.into_pyarray(py)
            .reshape([n_genes, n_groups])?
            .into_any(),
        hvg.into_pyarray(py).into_any(),
    ))
}

/// PCA on the residuals of a meta cell matrix
///
/// The twin of `rs_mc_pca_residuals`.
///
/// ### Params
///
/// * `data` - CSR raw counts
/// * `indices` - CSR column indices
/// * `indptr` - CSR row pointers
/// * `shape` - `(n_metacells, n_genes)`
/// * `fit` - A fit from [mc_fit_residuals]
/// * `gene_indices` - 0-based genes, all covered by the model
/// * `no_pcs` - Number of components
/// * `params` - `to_rust(PcaParams)` with `clr` and `normalise_variance` off
/// * `seed` - Seed for the randomised SVD
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(scores, loadings, singular_values)`, `float32`.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, shape, fit, gene_indices, no_pcs, params, seed, verbose = false
))]
#[allow(clippy::too_many_arguments)]
pub fn mc_pca_residuals<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
    shape: (usize, usize),
    fit: &PyResidualFit,
    gene_indices: PyReadonlyArray1<'py, u32>,
    no_pcs: usize,
    params: &Bound<'py, PyAny>,
    seed: usize,
    verbose: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let params: SingleCellPcaParams = from_py(params, "pca params")?;
    let m = csr_from_parts((&data, &indices, &indptr, shape), |x| x as u32)?;
    let genes = index_vec(&gene_indices)?;
    let (scores, loadings, s) = py
        .detach(|| {
            pca_residuals_in_memory(
                &m,
                &fit.inner,
                &genes,
                no_pcs,
                &params,
                seed,
                verbose as usize,
            )
        })
        .map_err(BvErr::from)?;
    Ok((
        mat_array(py, &scores)?,
        mat_array(py, &loadings)?,
        s.into_pyarray(py).into_any(),
    ))
}
