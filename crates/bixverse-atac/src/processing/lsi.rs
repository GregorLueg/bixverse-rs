//! TF-IDF and latent semantic indexing (LSI) on an ATAC store.
//!
//! The TF-IDF is store metadata (`store::v5::set_tf_idf`): the norm layer of a
//! TF-IDF store reads as `ln1p(x / lib * scale * idf)`, Signac `RunTFIDF`
//! method 1. LSI is then an uncentred, unscaled truncated SVD of that layer
//! over the top features, with the left singular vectors z-scored per
//! component (Signac `RunSVD`). The SVD reuses the single cell PCA solvers,
//! so the matrix is read sparse and never densified.
//!
//! The first component usually tracks sequencing depth. It is not dropped
//! here; [`LsiResult::depth_cor`] reports the correlation so the caller can.
//!
//! New cells project onto a fitted LSI as `X V diag(1/d)` ([`project_lsi`]),
//! Seurat's `ProjectSVD(mode = "lsi")`. The TF-IDF of the projected cells can
//! use any IDF ([`read_tf_idf`]): the reference's, or their own as Seurat does.

use faer::{Mat, MatRef};
use indexmap::IndexSet;
use rayon::prelude::*;

use bixverse_core::core::math::vector_helpers::pearson_correlation;
use bixverse_sc::sc_processing::pca::{PcaSolver, SingleCellPcaParams, sparse_pca_on_matrix};
use bixverse_sc_data::store::v5::{NormKind, set_tf_idf, tf_idf_weights};
use bixverse_sc_data::store::{chunk::norm_value, reader::peek_target_size};

use crate::prelude::*;

/// TF-IDF scale factor of Signac and muon.
pub const TF_IDF_SCALE: f32 = 1e4;

///////////
// Types //
///////////

/// Parameters of the LSI.
#[derive(Clone, Copy, Debug)]
pub struct LsiParams {
    /// Number of components. Clamped to `n_cells - 1` as Signac does.
    pub n_comp: usize,
    /// SVD solver, see [`PcaSolver`].
    pub solver: PcaSolver,
    /// Z-score each component over the cells (Signac `scale.embeddings`).
    pub scale_embeddings: bool,
    /// Clip the z-scored embedding to `+-scale_max` (Signac `scale.max`).
    pub scale_max: Option<f32>,
}

impl LsiParams {
    /// Generate a new instance of [`LsiParams`].
    ///
    /// ### Params
    ///
    /// * `n_comp` - Number of components
    /// * `solver` - SVD solver
    /// * `scale_embeddings` - Z-score each component over the cells
    /// * `scale_max` - Optional clip of the z-scored embedding
    ///
    /// ### Returns
    ///
    /// Initialised self.
    pub fn new(
        n_comp: usize,
        solver: PcaSolver,
        scale_embeddings: bool,
        scale_max: Option<f32>,
    ) -> Self {
        Self {
            n_comp,
            solver,
            scale_embeddings,
            scale_max,
        }
    }
}

impl Default for LsiParams {
    /// Signac's `RunSVD` defaults: 50 components, scaled, no clip.
    fn default() -> Self {
        Self {
            n_comp: 50,
            solver: PcaSolver::Auto,
            scale_embeddings: true,
            scale_max: None,
        }
    }
}

/// Result of [`lsi_on_store`].
#[derive(Clone, Debug)]
pub struct LsiResult {
    /// Cells x components embedding, rows in the order of the input cells.
    /// Z-scored per component unless `scale_embeddings` was off, in which
    /// case it is the left singular vectors.
    pub embedding: Mat<f32>,
    /// Features x components right singular vectors, rows in the order of
    /// the input features.
    pub loadings: Mat<f32>,
    /// Singular values, descending.
    pub singular_values: Vec<f32>,
    /// Pearson correlation of each component with the library size of the
    /// cells (Signac `DepthCor`). `NaN` for a constant component.
    pub depth_cor: Vec<f32>,
}

///////////////
// Functions //
///////////////

/// Switch a store to TF-IDF normalisation over a set of cells.
///
/// Computes the IDF over `cells` and writes it into both files of the store,
/// see [`set_tf_idf`]. Call again after the cell set changes.
///
/// ### Params
///
/// * `cell_path` - Cell-based file of the store
/// * `gene_path` - Gene-based (feature-major) file of the store
/// * `cells` - Cells the IDF is computed over, `None` for all
/// * `scale` - Scale factor, [`TF_IDF_SCALE`] for Signac and muon
///
/// ### Returns
///
/// The IDF, one weight per feature in store order.
pub fn set_tf_idf_on_store(
    cell_path: &str,
    gene_path: &str,
    cells: Option<&[usize]>,
    scale: f32,
) -> Result<Vec<f32>, AtacErrors> {
    let idf = tf_idf_weights(&ParallelSparseReader::new(gene_path)?, cells)?;
    set_tf_idf(&[cell_path, gene_path], &idf, scale)?;
    Ok(idf)
}

/// Mean and sample standard deviation (n - 1) of a column.
///
/// ### Params
///
/// * `x` - The values
///
/// ### Returns
///
/// `(mean, sd)` in f64; `sd` is 0 for fewer than two values.
fn mean_sd(x: &[f32]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    if x.len() < 2 {
        return (mean, 0.0);
    }
    let ss: f64 = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
    (mean, (ss / (n - 1.0)).sqrt())
}

/// TF-IDF of a block of a store, cells x features.
///
/// With `idf = None` this is the store's own norm layer (the store must be
/// TF-IDF). With an explicit IDF the values are recomputed from the raw counts
/// as `ln1p(x / lib * scale * idf_f)` through the same [`norm_value`], with the
/// store's scale and library sizes; passing the store's own IDF gives the norm
/// layer bit for bit.
///
/// ### Params
///
/// * `cell_path` - Cell-based file of the store (for the library sizes)
/// * `gene_path` - Gene-based (feature-major) file of the store
/// * `cells` - Cells, in output row order, no duplicates
/// * `features` - Features, in output column order
/// * `idf` - One weight per store feature, or `None` for the store's IDF
///
/// ### Returns
///
/// A CSR matrix with the raw counts in `data` and the TF-IDF in `data_2`.
pub fn read_tf_idf(
    cell_path: &str,
    gene_path: &str,
    cells: &[usize],
    features: &[usize],
    idf: Option<&[f32]>,
) -> Result<CompressedSparseData2<f32>, AtacErrors> {
    let gene_reader = ParallelSparseReader::new(gene_path)?;
    let cell_set: IndexSet<u32> = cells.iter().map(|&c| c as u32).collect();
    if cell_set.len() != cells.len() {
        return Err(AtacErrors::InvalidArgument(
            "read_tf_idf: duplicated cells".into(),
        ));
    }
    let Some(idf) = idf else {
        if gene_reader.norm_kind() != NormKind::TfIdf {
            return Err(AtacErrors::NotTfIdf);
        }
        let layers = DataLayerReturn::BothLayers;
        return Ok(gene_reader.read_genes_sparse_csr(features, Some(&cell_set), &layers)?);
    };
    if idf.len() != gene_reader.dims().total_genes {
        return Err(AtacErrors::InvalidArgument(format!(
            "IDF has {} weights for {} features",
            idf.len(),
            gene_reader.dims().total_genes
        )));
    }
    let scale = peek_target_size(gene_path)?
        .ok_or_else(|| AtacErrors::InvalidArgument("the store records no target size".into()))?;
    let lib = library_sizes(cell_path, cells)?;

    let mut m: CompressedSparseData2<f32> =
        gene_reader.read_genes_sparse_csr(features, Some(&cell_set), &DataLayerReturn::Raw)?;
    let (indptr, indices, raw, lib) = (&m.indptr, &m.indices, &m.data, &lib);
    let tf_idf: Vec<f32> = (0..cells.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            (indptr[i] as usize..indptr[i + 1] as usize).map(move |p| {
                let w = scale * idf[features[indices[p] as usize]];
                norm_value(raw[p], lib[i], w)
            })
        })
        .collect();
    m.data_2 = Some(tf_idf);
    Ok(m)
}

/// Project cells onto a fitted LSI.
///
/// `X V diag(1/d)`, Seurat's `ProjectSVD(mode = "lsi")` without centring or
/// scaling. A cell of the fit itself projects onto its left singular vector,
/// before the z-scoring of [`LsiResult::embedding`].
///
/// ### Params
///
/// * `tf_idf` - Cells x features CSR, TF-IDF in `data_2`, columns in the row
///   order of `loadings`, see [`read_tf_idf`]
/// * `loadings` - Features x components, [`LsiResult::loadings`]
/// * `singular_values` - [`LsiResult::singular_values`]
///
/// ### Returns
///
/// Cells x components.
pub fn project_lsi(
    tf_idf: &CompressedSparseData2<f32>,
    loadings: MatRef<f32>,
    singular_values: &[f32],
) -> Result<Mat<f32>, AtacErrors> {
    let n_comp = loadings.ncols();
    let values = tf_idf.data_2.as_ref().ok_or_else(|| {
        AtacErrors::InvalidArgument("project_lsi needs the TF-IDF in data_2".into())
    })?;
    if !tf_idf.cs_type.is_csr() || tf_idf.shape.1 != loadings.nrows() {
        return Err(AtacErrors::InvalidArgument(format!(
            "project_lsi needs a CSR with {} columns, got {:?}",
            loadings.nrows(),
            tf_idf.shape
        )));
    }
    if singular_values.len() < n_comp {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} singular values for {n_comp} components",
            singular_values.len()
        )));
    }
    let n_cells = tf_idf.shape.0;
    let mut rows = vec![0.0f32; n_cells * n_comp];
    rows.par_chunks_mut(n_comp.max(1))
        .enumerate()
        .for_each(|(i, row)| {
            let (lo, hi) = (tf_idf.indptr[i] as usize, tf_idf.indptr[i + 1] as usize);
            for (&f, &x) in tf_idf.indices[lo..hi].iter().zip(&values[lo..hi]) {
                for (j, r) in row.iter_mut().enumerate() {
                    *r += x * loadings[(f as usize, j)];
                }
            }
            for (r, &d) in row.iter_mut().zip(singular_values) {
                *r /= d;
            }
        });
    Ok(Mat::from_fn(n_cells, n_comp, |i, j| rows[i * n_comp + j]))
}

/// LSI of a TF-IDF store.
///
/// ### Params
///
/// * `cell_path` - Cell-based file of the store (for the library sizes)
/// * `gene_path` - Gene-based (feature-major) file of the store
/// * `cells` - Cells to embed, in output row order
/// * `features` - Features to use, typically from
///   [`crate::processing::features::select_top_features`]
/// * `params` - See [`LsiParams`]
/// * `seed` - Seed of the solver
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`LsiResult`], or [`AtacErrors::NotTfIdf`] for a store that is not
/// TF-IDF normalised.
pub fn lsi_on_store(
    cell_path: &str,
    gene_path: &str,
    cells: &[usize],
    features: &[usize],
    params: &LsiParams,
    seed: usize,
    verbose: usize,
) -> Result<LsiResult, AtacErrors> {
    let tf_idf = read_tf_idf(cell_path, gene_path, cells, features, None)?;
    let depth = library_sizes(cell_path, cells)?;
    lsi_on_matrix(tf_idf, &depth, params, seed, verbose)
}

/// Library sizes of cells as `f32`.
///
/// ### Params
///
/// * `cell_path` - Cell-based file of the store
/// * `cells` - The cells
///
/// ### Returns
///
/// One library size per cell.
pub(crate) fn library_sizes(cell_path: &str, cells: &[usize]) -> Result<Vec<f32>, AtacErrors> {
    Ok(ParallelSparseReader::new(cell_path)?
        .read_cell_library_sizes(cells)?
        .into_iter()
        .map(|x| x as f32)
        .collect())
}

/// LSI of a TF-IDF block, see [`lsi_on_store`].
///
/// ### Params
///
/// * `tf_idf` - Cells x features CSR from [`read_tf_idf`], TF-IDF in `data_2`
/// * `depth` - Library size per row, for [`LsiResult::depth_cor`]
/// * `params` - See [`LsiParams`]
/// * `seed` - Seed of the solver
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// The [`LsiResult`].
pub fn lsi_on_matrix(
    tf_idf: CompressedSparseData2<f32>,
    depth: &[f32],
    params: &LsiParams,
    seed: usize,
    verbose: usize,
) -> Result<LsiResult, AtacErrors> {
    let n_cells = tf_idf.shape.0;
    if n_cells < 2 || tf_idf.shape.1 == 0 {
        return Err(AtacErrors::InvalidArgument(format!(
            "LSI needs at least two cells and one feature, got {} and {}",
            n_cells, tf_idf.shape.1
        )));
    }
    if depth.len() != n_cells {
        return Err(AtacErrors::InvalidArgument(format!(
            "{} library sizes for {n_cells} cells",
            depth.len()
        )));
    }
    let n_comp = params.n_comp.min(n_cells - 1);

    let pca_params = SingleCellPcaParams::new(false, false, params.solver, false, 1.0);
    let (scores, loadings, singular_values, _, _) =
        sparse_pca_on_matrix(tf_idf, n_comp, &pca_params, None, seed, verbose)?;

    // scores are U * diag(s); z-scoring makes the scale irrelevant, so only
    // the unscaled path divides it back out
    let columns: Vec<Vec<f32>> = (0..n_comp)
        .into_par_iter()
        .map(|j| {
            let col: Vec<f32> = (0..scores.nrows()).map(|i| scores[(i, j)]).collect();
            if params.scale_embeddings {
                let (mean, sd) = mean_sd(&col);
                col.iter()
                    .map(|&v| {
                        let z = ((v as f64 - mean) / sd) as f32;
                        match params.scale_max {
                            Some(m) => z.clamp(-m, m),
                            None => z,
                        }
                    })
                    .collect()
            } else {
                let s = singular_values[j];
                col.iter().map(|&v| v / s).collect()
            }
        })
        .collect();

    let depth_cor: Vec<f32> = columns
        .par_iter()
        .map(|col| pearson_correlation(col, depth).map_or(f32::NAN, |r| r as f32))
        .collect();

    let embedding = Mat::from_fn(n_cells, n_comp, |i, j| columns[j][i]);

    Ok(LsiResult {
        embedding,
        loadings,
        singular_values,
        depth_cor,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use bixverse_sc_data::store::gene_file::write_gene_file;
    use bixverse_sc_data::store::writer::CellGeneSparseWriter;

    use crate::processing::features::{TopFeatureCutoff, feature_total_stats, select_top_features};

    /// Scratch files that remove themselves on drop.
    struct TempFiles(Vec<std::path::PathBuf>);

    impl TempFiles {
        /// Two paths (cell file, gene file) unique to the test.
        ///
        /// ### Params
        ///
        /// * `tag` - Test name
        ///
        /// ### Returns
        ///
        /// The guard.
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir();
            let pid = std::process::id();
            Self(vec![
                dir.join(format!("bv_atac_{tag}_{pid}_cells.bin")),
                dir.join(format!("bv_atac_{tag}_{pid}_genes.bin")),
            ])
        }

        /// Path as `&str`.
        ///
        /// ### Params
        ///
        /// * `i` - 0 cell file, 1 gene file
        ///
        /// ### Returns
        ///
        /// The path.
        fn path(&self, i: usize) -> &str {
            self.0[i].to_str().expect("temp path is valid UTF-8")
        }
    }

    impl Drop for TempFiles {
        fn drop(&mut self) {
            for p in &self.0 {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    /// Two groups of cells with different accessible feature blocks, plus a
    /// shared block, counts 1 or 2.
    ///
    /// ### Returns
    ///
    /// `dense[cell][feature]` counts.
    fn two_group_counts() -> Vec<Vec<u32>> {
        let (n_cells, n_features) = (40, 30);
        (0..n_cells)
            .map(|c| {
                (0..n_features)
                    .map(|f| {
                        let own = if c < 20 {
                            f < 12
                        } else {
                            (12..24).contains(&f)
                        };
                        let shared = f >= 24;
                        let hit = (own && (c * 7 + f * 3) % 3 != 0) || (shared && (c + f) % 4 == 0);
                        if hit {
                            1 + ((c + f) % 5 == 0) as u32
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Write both files of a store from dense counts.
    ///
    /// ### Params
    ///
    /// * `files` - Where to write
    /// * `dense` - `dense[cell][feature]` counts
    fn write_store(files: &TempFiles, dense: &[Vec<u32>]) {
        let mut writer =
            CellGeneSparseWriter::new(files.path(0), true, dense.len(), dense[0].len(), 1e4)
                .expect("writer opens");
        for (cell_idx, cell) in dense.iter().enumerate() {
            let (raw, indices): (Vec<u32>, Vec<u32>) = cell
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0)
                .map(|(g, &v)| (v, g as u32))
                .unzip();
            writer
                .write_cell_chunk(CsrCellChunk::from_data(&raw, &indices, cell_idx, 1e4, true))
                .expect("write cell chunk");
        }
        writer.finalise().expect("finalise");
        write_gene_file(files.path(0), files.path(1), None, false).expect("gene file");
    }

    /// Z-score one column (n - 1).
    ///
    /// ### Params
    ///
    /// * `x` - The values
    ///
    /// ### Returns
    ///
    /// The z-scores.
    fn zscore(x: &[f64]) -> Vec<f64> {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        x.iter().map(|v| (v - m) / sd).collect()
    }

    #[test]
    fn test_lsi_matches_dense_tfidf_svd() {
        let dense = two_group_counts();
        let (n_cells, n_features) = (dense.len(), dense[0].len());
        let files = TempFiles::new("lsi_dense");
        write_store(&files, &dense);

        let cells: Vec<usize> = (0..n_cells).collect();
        let idf =
            set_tf_idf_on_store(files.path(0), files.path(1), None, TF_IDF_SCALE).expect("tf-idf");
        let totals =
            feature_total_stats(&ParallelSparseReader::new(files.path(1)).unwrap(), None).unwrap();
        let features = select_top_features(&totals, TopFeatureCutoff::All);

        let n_comp = 4;
        let params = LsiParams::new(n_comp, PcaSolver::Exact, true, None);
        let res = lsi_on_store(
            files.path(0),
            files.path(1),
            &cells,
            &features,
            &params,
            42,
            0,
        )
        .expect("lsi");

        // dense reference: Signac RunTFIDF method 1, then a full SVD
        let lib: Vec<f64> = dense
            .iter()
            .map(|c| c.iter().map(|&v| v as f64).sum())
            .collect();
        let x = Mat::<f64>::from_fn(n_cells, n_features, |c, f| {
            let tf = dense[c][f] as f64 / lib[c];
            (tf * TF_IDF_SCALE as f64 * idf[f] as f64).ln_1p()
        });
        let svd = x.thin_svd().expect("svd");
        for j in 0..n_comp {
            assert_relative_eq!(
                res.singular_values[j] as f64,
                svd.S()[j],
                max_relative = 1e-3
            );
            let reference = zscore(&(0..n_cells).map(|i| svd.U()[(i, j)]).collect::<Vec<_>>());
            let ours: Vec<f64> = (0..n_cells).map(|i| res.embedding[(i, j)] as f64).collect();
            let sign = if reference.iter().zip(&ours).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
                -1.0
            } else {
                1.0
            };
            for (a, b) in reference.iter().zip(&ours) {
                assert_relative_eq!(a * sign, *b, epsilon = 1e-3);
            }
        }
        assert_eq!(res.depth_cor.len(), n_comp);
        assert!(res.depth_cor.iter().all(|r| r.abs() <= 1.0));
    }

    #[test]
    fn test_lsi_refuses_a_log_cpm_store() {
        let dense = two_group_counts();
        let files = TempFiles::new("lsi_logcpm");
        write_store(&files, &dense);
        let cells: Vec<usize> = (0..dense.len()).collect();
        let err = lsi_on_store(
            files.path(0),
            files.path(1),
            &cells,
            &[0, 1, 2],
            &LsiParams::default(),
            1,
            0,
        );
        assert!(matches!(err, Err(AtacErrors::NotTfIdf)));
    }

    #[test]
    fn test_mean_sd_uses_n_minus_one() {
        let (m, s) = mean_sd(&[1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(m, 2.5);
        assert_relative_eq!(s, (5.0f64 / 3.0).sqrt());
    }
}
