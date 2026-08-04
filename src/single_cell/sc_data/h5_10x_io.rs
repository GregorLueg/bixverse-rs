//! 10x CellRanger v2/v3 h5 ingestion. Streams the gene-expression modality
//! to the bixverse binarised format. Other modalities (e.g. Antibody Capture)
//! are filtered out here.

use hdf5::{
    Dataset, File,
    types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode},
};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_data::H5_CELL_SLICE_SIZE;
use crate::single_cell::sc_data::data_io::CellOnFileQuality;
use crate::single_cell::sc_data::data_io::*;

/////////////////
// 10x version //
/////////////////

/// CellRanger h5 layout. Both store genes x cells in CSC (indptr over cells,
/// indices = gene rows). V3 prefixes datasets with `matrix/`.
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum TenxVersion {
    /// CellRanger v2: root-level datasets, single modality
    V2,
    /// CellRanger v3+: `matrix/` prefix, may be multi-modal
    V3,
}

impl TenxVersion {
    /// Returns the indptr path
    ///
    /// ### Returns
    ///
    /// `indptr` path from the h5 file
    #[inline]
    pub fn get_indptr(&self) -> &str {
        match self {
            TenxVersion::V2 => "indptr",
            TenxVersion::V3 => "matrix/indptr",
        }
    }

    /// Returns the indices path
    ///
    /// ### Returns
    ///
    /// `indices` path from the h5 file
    #[inline]
    pub fn get_indices(&self) -> &str {
        match self {
            TenxVersion::V2 => "indices",
            TenxVersion::V3 => "matrix/indices",
        }
    }

    /// Returns the data path
    ///
    /// ### Returns
    ///
    /// `data` path from the h5 file
    #[inline]
    pub fn get_data(&self) -> &str {
        match self {
            TenxVersion::V2 => "data",
            TenxVersion::V3 => "matrix/data",
        }
    }

    /// Returns the shape path
    ///
    /// ### Returns
    ///
    /// `shape` path from the h5 file
    #[inline]
    pub fn get_shape(&self) -> &str {
        match self {
            TenxVersion::V2 => "shape",
            TenxVersion::V3 => "matrix/shape",
        }
    }

    /// Returns the feature_type path (V3 only)
    ///
    /// ### Returns
    ///
    /// `feature_type` path from the h5 file
    #[inline]
    pub fn get_feature_type(&self) -> Option<&str> {
        match self {
            TenxVersion::V2 => None,
            TenxVersion::V3 => Some("matrix/features/feature_type"),
        }
    }

    /// Returns the feature name path
    ///
    /// ### Returns
    ///
    /// `feature name` path from the h5 file
    #[inline]
    pub fn get_feature_name(&self) -> Option<&str> {
        match self {
            TenxVersion::V2 => Some("gene_names"),
            TenxVersion::V3 => Some("matrix/features/name"),
        }
    }

    /// Returns the barcodes path
    ///
    /// ### Returns
    ///
    /// `barcodes` path from the h5 file
    #[inline]
    pub fn get_barcodes(&self) -> &str {
        match self {
            TenxVersion::V2 => "barcodes",
            TenxVersion::V3 => "matrix/barcodes",
        }
    }
}

/// Read a fixed-length string dataset into owned Strings
///
/// ### Params
///
/// * `ds` - The h5ad data set
fn read_fixed<T>(ds: &Dataset) -> hdf5::Result<Vec<String>>
where
    T: hdf5::H5Type + AsRef<str>,
{
    Ok(ds
        .read_raw::<T>()?
        .iter()
        .map(|s| s.as_ref().trim_end_matches('\0').trim().to_string())
        .collect())
}

/// Read a 1-D HDF5 string dataset regardless of on-disk encoding
///
/// CellRanger files written by h5py use variable-length UTF-8; older or
/// non-h5py files may use fixed-length or ASCII. This inspects the datatype
/// and dispatches to the matching Rust string type.
///
/// ### Params
///
/// * `ds` - The string dataset to read
///
/// ### Returns
///
/// The strings as a `Vec<String>`.
pub fn read_string_dataset(ds: &Dataset) -> Result<Vec<String>, BixverseErrors> {
    let strings = match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => ds
            .read_raw::<VarLenUnicode>()?
            .iter()
            .map(|s| s.as_str().to_string())
            .collect(),
        TypeDescriptor::VarLenAscii => ds
            .read_raw::<VarLenAscii>()?
            .iter()
            .map(|s| s.as_str().to_string())
            .collect(),
        TypeDescriptor::FixedUnicode(n) => match n {
            ..=15 => read_fixed::<FixedUnicode<15>>(ds)?,
            16..=31 => read_fixed::<FixedUnicode<31>>(ds)?,
            _ => read_fixed::<FixedUnicode<63>>(ds)?,
        },
        TypeDescriptor::FixedAscii(n) => match n {
            ..=15 => read_fixed::<FixedAscii<15>>(ds)?,
            16..=31 => read_fixed::<FixedAscii<31>>(ds)?,
            _ => read_fixed::<FixedAscii<63>>(ds)?,
        },
        other => {
            return Err(BixverseErrors::H5UnexpectedStringType(format!(
                "{:?}",
                other
            )));
        }
    };

    Ok(strings)
}

/// Parse a 10x version string
///
/// ### Params
///
/// * `s` - String to parse
///
/// ### Returns
///
/// An option of the [TenxVersion].
pub fn parse_tenx_version(s: &str) -> Option<TenxVersion> {
    match s.to_lowercase().as_str() {
        "v2" => Some(TenxVersion::V2),
        "v3" => Some(TenxVersion::V3),
        _ => None,
    }
}

/// Read the matrix dimensions from a 10x file
///
/// 10x stores shape as `[no_genes, no_cells]`.
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
/// * `version` - The optional [TenxVersion]
///
/// ### Returns
///
/// A tuple `(no_cells, no_genes)` in bixverse convention.
pub fn get_tenx_dimensions<P: AsRef<Path>>(
    file_path: P,
    version: Option<TenxVersion>,
) -> Result<(usize, usize), BixverseErrors> {
    let version = resolve_tenx_version(&file_path, version)?;
    let file = File::open(file_path)?;
    let shape: Vec<usize> = file.dataset(version.get_shape())?.read_1d()?.to_vec();
    Ok((shape[1], shape[0]))
}

/// Auto-detect the 10x version from the on-disk layout
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
///
/// ### Returns
///
/// The detected [TenxVersion], or an error if neither layout matches.
pub fn detect_tenx_version<P: AsRef<Path>>(file_path: P) -> Result<TenxVersion, BixverseErrors> {
    let file = File::open(file_path)?;
    if file.dataset("matrix/data").is_ok() {
        Ok(TenxVersion::V3)
    } else if file.dataset("data").is_ok() && file.dataset("genes").is_ok() {
        Ok(TenxVersion::V2)
    } else {
        Err(hdf5::Error::from(
            "could not auto-detect 10x version: neither 'matrix/data' (v3) nor root 'data'+'genes' (v2) found",
        )
        .into())
    }
}

/// Resolve a user-supplied version, falling back to auto-detection
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
/// * `version` - An option of the [TenxVersion]
///
/// ### Returns
///
/// The final tenx version
pub fn resolve_tenx_version<P: AsRef<Path>>(
    file_path: P,
    version: Option<TenxVersion>,
) -> Result<TenxVersion, BixverseErrors> {
    match version {
        Some(v) => Ok(v),
        None => detect_tenx_version(file_path),
    }
}

/////////////
// Writers //
/////////////

/// Streams 10x gene-expression counts to disk in the binarised format
///
/// Filters to the gene-expression modality (V3), applies the QC thresholds and
/// streams cells directly to CSR on disk. Since 10x stores cells along the
/// indptr axis already, no transpose is required.
///
/// ### Params
///
/// * `h5_path` - Path to the 10x h5 file.
/// * `bin_path` - Path to the binarised object on disk to write to.
/// * `version` - The option of the [TenxVersion]. If not provided, it will
///   attempt automated detection.
/// * `no_cells` - Total number of cells (columns) in the data.
/// * `no_genes` - Total number of features (rows), incl. non-gene modalities.
/// * `cell_quality` - Minimum cell/gene quality + target size for
///   normalisation.
/// * `feature_type` - Target modality for V3. Defaults to "Gene Expression".
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// A tuple with `(no_cells, no_genes, cell quality metrics)`.
#[allow(clippy::too_many_arguments)]
pub fn stream_h5_tenx_counts<P: AsRef<Path>>(
    h5_path: P,
    bin_path: P,
    version: Option<TenxVersion>,
    no_cells: usize,
    no_genes: usize,
    cell_quality: MinCellQuality,
    feature_type: Option<&str>,
    verbose: bool,
) -> Result<(usize, usize, CellQuality), BixverseErrors> {
    let version = resolve_tenx_version(&h5_path, version)?;

    if verbose {
        println!("Step 1/3: Analysing data structure and calculating QC metrics...");
    }

    let file_quality = parse_h5_tenx_quality(
        &h5_path,
        version,
        (no_cells, no_genes),
        &cell_quality,
        feature_type,
        verbose,
    )?;

    if verbose {
        println!("Step 2/3: QC Results:");
        println!(
            "  Genes passing QC: {} / {}",
            file_quality.genes_to_keep.len().separate_with_underscores(),
            no_genes.separate_with_underscores()
        );
        println!(
            "  Cells passing QC: {} / {}",
            file_quality.cells_to_keep.len().separate_with_underscores(),
            no_cells.separate_with_underscores()
        );
        println!("Step 3/3: Writing cells to CSR format...");
    }

    let mut cell_qc = write_h5_tenx_streaming(
        &h5_path,
        &bin_path,
        version,
        &file_quality,
        cell_quality,
        verbose,
    )?;

    cell_qc.set_cell_indices(&file_quality.cells_to_keep);
    cell_qc.set_gene_indices(&file_quality.genes_to_keep);

    Ok((
        file_quality.cells_to_keep.len(),
        file_quality.genes_to_keep.len(),
        cell_qc,
    ))
}

//////////////////
// QC functions //
//////////////////

/// Validate and filter feature types for 10x V3 files
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
/// * `version` - The [TenxVersion] (expects V3)
/// * `target_feature_type` - Modality to keep, defaults to "Gene Expression"
///
/// ### Returns
///
/// Row indices of the features matching the target modality.
pub fn validate_feature_types_tenx<P: AsRef<Path>>(
    file_path: P,
    version: TenxVersion,
    target_feature_type: Option<&str>,
) -> Result<Vec<usize>, BixverseErrors> {
    let file = File::open(file_path)?;
    let ft_path = version
        .get_feature_type()
        .expect("validate_feature_types_tenx called on a version without feature types");

    let feature_type_ds = file.dataset(ft_path)?;
    let feature_types: Vec<String> = read_string_dataset(&feature_type_ds)?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();

    let unique_types: FxHashSet<&str> = feature_types.iter().map(|s| s.as_str()).collect();

    // single modality, nothing to filter
    if unique_types.len() == 1 {
        return Ok((0..feature_types.len()).collect());
    }

    let target = target_feature_type.unwrap_or("Gene Expression");

    if !unique_types.contains(target) {
        return Err(BixverseErrors::FeatureTypeNotFound {
            requested: target.to_string(),
            found: unique_types.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    Ok(feature_types
        .iter()
        .enumerate()
        .filter(|(_, ft)| ft.as_str() == target)
        .map(|(i, _)| i)
        .collect())
}

/// Get the cell quality data from a 10x file
///
/// Two passes over the cells: first counting how many cells express each gene
/// (for gene filtering), then computing per-cell unique genes and library size
/// over the kept genes only. Non-gene modalities are excluded because they are
/// never part of `genes_to_keep`.
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
/// * `version` - The [TenxVersion]
/// * `shape` - Tuple with `(no_cells, no_genes)` (genes incl. all modalities).
/// * `cell_quality` - Minimum quality thresholds.
/// * `feature_type` - Target modality for V3.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `CellOnFileQuality` describing which cells and genes to include.
pub fn parse_h5_tenx_quality<P: AsRef<Path>>(
    file_path: P,
    version: TenxVersion,
    shape: (usize, usize),
    cell_quality: &MinCellQuality,
    feature_type: Option<&str>,
    verbose: bool,
) -> Result<CellOnFileQuality, BixverseErrors> {
    let file_path = file_path.as_ref();
    let (n_cells, n_genes) = shape;

    if verbose {
        println!(
            "  Reading 10x matrix structure (shape: {} cells x {} features)...",
            n_cells.separate_with_underscores(),
            n_genes.separate_with_underscores()
        );
    }

    let file = File::open(file_path)?;
    let indptr: Vec<usize> = file.dataset(version.get_indptr())?.read_1d()?.to_vec();

    let feature_indices = match version {
        TenxVersion::V3 => validate_feature_types_tenx(file_path, version, feature_type)?,
        TenxVersion::V2 => (0..n_genes).collect(),
    };

    if verbose && version == TenxVersion::V3 {
        println!(
            "  Features after type filtering: {} / {}",
            feature_indices.len().separate_with_underscores(),
            n_genes.separate_with_underscores()
        );
    }

    const CELL_CHUNK_SIZE: usize = 10000;
    let chunks: Vec<usize> = (0..n_cells).step_by(CELL_CHUNK_SIZE).collect();
    let num_chunks = chunks.len();
    let report_interval = (num_chunks / 10).max(1);

    if verbose {
        println!("First pass - gene expression statistics:");
    }

    let first_pass_time = Instant::now();
    let completed_chunks = Arc::new(AtomicUsize::new(0));

    let gene_counts: Vec<Vec<usize>> = chunks
        .par_iter()
        .map(|&chunk_start_cell| {
            let mut local_counts = vec![0usize; n_genes];

            let Ok(file) = File::open(file_path) else {
                return local_counts;
            };
            let Ok(indices_ds) = file.dataset(version.get_indices()) else {
                return local_counts;
            };

            let chunk_end_cell = (chunk_start_cell + CELL_CHUNK_SIZE).min(n_cells) - 1;
            let data_start = indptr[chunk_start_cell];
            let data_end = indptr[chunk_end_cell + 1];

            if data_start >= data_end {
                return local_counts;
            }

            let Ok(chunk_indices) = indices_ds.read_slice_1d(data_start..data_end) else {
                return local_counts;
            };
            let chunk_indices: Vec<i64> = chunk_indices.to_vec();

            for cell_idx in chunk_start_cell..=chunk_end_cell {
                let cell_data_start = indptr[cell_idx] - data_start;
                let cell_data_end = indptr[cell_idx + 1] - data_start;

                for local_idx in cell_data_start..cell_data_end {
                    let gene_idx = chunk_indices[local_idx] as usize;
                    if gene_idx < n_genes {
                        local_counts[gene_idx] += 1;
                    }
                }
            }

            if verbose {
                let completed = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                if completed.is_multiple_of(report_interval) || completed == num_chunks {
                    let progress =
                        ((completed as f64 / num_chunks as f64 * 10.0).round() as usize) * 10;
                    println!(
                        "  Processed {}% of chunks ({}/{})",
                        progress, completed, num_chunks
                    );
                }
            }

            local_counts
        })
        .collect();

    let mut no_cells_exp_gene = vec![0usize; n_genes];
    for local_counts in gene_counts {
        for (i, count) in local_counts.into_iter().enumerate() {
            no_cells_exp_gene[i] += count;
        }
    }

    let genes_to_keep: Vec<usize> = feature_indices
        .iter()
        .copied()
        .filter(|&g| no_cells_exp_gene[g] >= cell_quality.min_cells)
        .collect();

    if verbose {
        println!("First pass done: {:.2?}", first_pass_time.elapsed());
        println!(
            "  Genes passing filter: {} / {}",
            genes_to_keep.len().separate_with_underscores(),
            feature_indices.len().separate_with_underscores()
        );
        println!("Second pass - cell statistics:");
    }

    let mut genes_to_keep_lookup = vec![false; n_genes];
    for &gene_idx in &genes_to_keep {
        genes_to_keep_lookup[gene_idx] = true;
    }

    let second_pass_time = Instant::now();
    let completed_chunks = Arc::new(AtomicUsize::new(0));

    let cell_stats: Vec<(Vec<usize>, Vec<f32>)> = chunks
        .par_iter()
        .map(|&chunk_start_cell| {
            let chunk_end_cell = (chunk_start_cell + CELL_CHUNK_SIZE).min(n_cells) - 1;
            let mut local_unique = vec![0usize; chunk_end_cell - chunk_start_cell + 1];
            let mut local_lib_size = vec![0.0f32; chunk_end_cell - chunk_start_cell + 1];

            let Ok(file) = File::open(file_path) else {
                return (local_unique, local_lib_size);
            };
            let (Ok(data_ds), Ok(indices_ds)) = (
                file.dataset(version.get_data()),
                file.dataset(version.get_indices()),
            ) else {
                return (local_unique, local_lib_size);
            };

            let data_start = indptr[chunk_start_cell];
            let data_end = indptr[chunk_end_cell + 1];

            if data_start >= data_end {
                return (local_unique, local_lib_size);
            }

            let (Ok(chunk_data), Ok(chunk_indices)) = (
                data_ds.read_slice_1d(data_start..data_end),
                indices_ds.read_slice_1d(data_start..data_end),
            ) else {
                return (local_unique, local_lib_size);
            };

            let chunk_data: Vec<f32> = chunk_data.to_vec();
            let chunk_indices: Vec<usize> = chunk_indices.to_vec();

            for cell_idx in chunk_start_cell..=chunk_end_cell {
                let cell_data_start = indptr[cell_idx] - data_start;
                let cell_data_end = indptr[cell_idx + 1] - data_start;
                let local_cell_idx = cell_idx - chunk_start_cell;

                for local_idx in cell_data_start..cell_data_end {
                    let gene_idx = chunk_indices[local_idx];
                    if genes_to_keep_lookup[gene_idx] {
                        local_unique[local_cell_idx] += 1;
                        local_lib_size[local_cell_idx] += chunk_data[local_idx];
                    }
                }
            }

            if verbose {
                let completed = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                if completed.is_multiple_of(report_interval) || completed == num_chunks {
                    let progress =
                        ((completed as f64 / num_chunks as f64 * 10.0).round() as usize) * 10;
                    println!(
                        "  Processed {}% of chunks ({}/{})",
                        progress, completed, num_chunks
                    );
                }
            }

            (local_unique, local_lib_size)
        })
        .collect();

    let mut cell_unique_genes = vec![0usize; n_cells];
    let mut cell_lib_size = vec![0.0f32; n_cells];

    for (chunk_idx, (local_unique, local_lib)) in cell_stats.into_iter().enumerate() {
        let chunk_start = chunks[chunk_idx];
        for (i, (unique, lib)) in local_unique.into_iter().zip(local_lib).enumerate() {
            cell_unique_genes[chunk_start + i] = unique;
            cell_lib_size[chunk_start + i] = lib;
        }
    }

    let cells_to_keep: Vec<usize> = (0..n_cells)
        .filter(|&i| {
            cell_unique_genes[i] >= cell_quality.min_unique_genes
                && cell_lib_size[i] >= cell_quality.min_lib_size as f32
        })
        .collect();

    if verbose {
        println!("Second pass done: {:.2?}", second_pass_time.elapsed());
        println!(
            "  Cells passing filter: {} / {}",
            cells_to_keep.len().separate_with_underscores(),
            n_cells.separate_with_underscores()
        );
    }

    let mut file_quality_data = CellOnFileQuality::new(cells_to_keep, genes_to_keep);
    file_quality_data.generate_maps_sets();

    Ok(file_quality_data)
}

/// Stream 10x cells directly to disk with batched reading
///
/// 10x stores cells along the indptr axis, so each cell's gene entries are
/// contiguous and can be normalised and written per-cell, like the h5ad CSR
/// path.
///
/// ### Params
///
/// * `file_path` - Path to the h5 file
/// * `bin_path` - Path to the binary file to write to
/// * `version` - The [TenxVersion]
/// * `quality` - Which cells and genes to include after the first pass.
/// * `cell_qc` - Minimum criteria and target size.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// `CellQuality` with NNZ and lib size per cell (indices set by the caller).
pub fn write_h5_tenx_streaming<P: AsRef<Path>>(
    file_path: P,
    bin_path: P,
    version: TenxVersion,
    quality: &CellOnFileQuality,
    cell_qc: MinCellQuality,
    verbose: bool,
) -> Result<CellQuality, BixverseErrors> {
    let file = File::open(&file_path)?;
    let data_ds = file.dataset(version.get_data())?;
    let indices_ds = file.dataset(version.get_indices())?;
    let indptr_ds = file.dataset(version.get_indptr())?;
    let indptr_raw: Vec<usize> = indptr_ds.read_1d()?.to_vec();

    let mut writer = CellGeneSparseWriter::new(
        bin_path,
        true,
        quality.cells_to_keep.len(),
        quality.genes_to_keep.len(),
        cell_qc.target_size,
    )?;

    let mut lib_size = Vec::with_capacity(quality.cells_to_keep.len());
    let mut nnz = Vec::with_capacity(quality.cells_to_keep.len());

    let total_cells = quality.cells_to_keep.len();
    let num_batches = total_cells.div_ceil(H5_CELL_SLICE_SIZE);

    if verbose {
        println!(
            "  Processing {} cells in batches of {}...",
            total_cells.separate_with_underscores(),
            H5_CELL_SLICE_SIZE.separate_with_underscores()
        );
    }

    let start_write = Instant::now();

    let mut cell_data: Vec<(usize, u32)> = Vec::with_capacity(10000);
    let mut gene_indices: Vec<u32> = Vec::with_capacity(10000);
    let mut gene_counts: Vec<u32> = Vec::with_capacity(10000);

    for (batch_idx, cell_batch) in quality.cells_to_keep.chunks(H5_CELL_SLICE_SIZE).enumerate() {
        if verbose && (batch_idx % ((num_batches / 10).max(1)) == 0 || batch_idx == num_batches - 1)
        {
            let progress = ((batch_idx as f64 / num_batches as f64 * 10.0).round() as usize) * 10;
            let processed = (batch_idx + 1) * H5_CELL_SLICE_SIZE;
            println!(
                "  Processed {}% ({} / {} cells)",
                progress,
                processed.min(total_cells).separate_with_underscores(),
                total_cells.separate_with_underscores()
            );
        }

        let start_pos = cell_batch.iter().map(|&c| indptr_raw[c]).min().unwrap_or(0);
        let end_pos = cell_batch
            .iter()
            .map(|&c| indptr_raw[c + 1])
            .max()
            .unwrap_or(0);

        if start_pos >= end_pos {
            for &old_cell_idx in cell_batch {
                lib_size.push(0);
                nnz.push(0);
                let new_cell_idx = quality.cell_old_to_new[&old_cell_idx];
                let empty_chunk = CsrCellChunk::from_data(
                    &[] as &[u32],
                    &[] as &[u32],
                    new_cell_idx,
                    cell_qc.target_size,
                    true,
                );
                writer.write_cell_chunk(empty_chunk)?;
            }
            continue;
        }

        let chunk_data: Vec<f32> = data_ds.read_slice_1d(start_pos..end_pos)?.to_vec();
        let chunk_indices: Vec<usize> = indices_ds.read_slice_1d(start_pos..end_pos)?.to_vec();

        for &old_cell_idx in cell_batch {
            let cell_start = indptr_raw[old_cell_idx];
            let cell_end = indptr_raw[old_cell_idx + 1];

            cell_data.clear();
            gene_indices.clear();
            gene_counts.clear();

            for idx in cell_start..cell_end {
                let local_idx = idx - start_pos;
                let old_gene_idx = chunk_indices[local_idx];

                if let Some(&new_gene_idx) = quality.gene_old_to_new.get(&old_gene_idx) {
                    let raw_val = chunk_data[local_idx] as u32;
                    cell_data.push((new_gene_idx, raw_val));
                }
            }

            if !cell_data.is_empty() {
                let needs_sort = cell_data.windows(2).any(|w| w[0].0 > w[1].0);
                if needs_sort {
                    cell_data.sort_unstable_by_key(|&(gene_idx, _)| gene_idx);
                }
                gene_indices.extend(cell_data.iter().map(|(g, _)| *g as u32));
                gene_counts.extend(cell_data.iter().map(|(_, c)| *c));
            }

            let new_cell_idx = quality.cell_old_to_new[&old_cell_idx];
            let cell_chunk = CsrCellChunk::from_data(
                &gene_counts,
                &gene_indices,
                new_cell_idx,
                cell_qc.target_size,
                true,
            );

            let (nnz_i, lib_size_i) = cell_chunk.get_qc_info();
            nnz.push(nnz_i);
            lib_size.push(lib_size_i);

            writer.write_cell_chunk(cell_chunk)?;
        }
    }

    writer.finalise()?;

    if verbose {
        println!("  Writing complete in {:.2?}", start_write.elapsed());
    }

    Ok(CellQuality {
        cell_indices: Vec::new(),
        gene_indices: Vec::new(),
        lib_size,
        nnz,
    })
}
