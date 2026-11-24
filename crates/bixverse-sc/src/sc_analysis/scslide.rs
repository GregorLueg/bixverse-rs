//! scSLIDE: one vector per sample from landmark densities, after Jiang et
//! al. (Cell 2026) and Seurat's `PrepareSampleObject` / `GenerateSampleObject`.
//!
//! A per-donor sketch trains a PLS model against a sample-level phenotype, every
//! cell is projected onto it, and a fixed-weight WNN between all cells and a
//! leverage-sampled set of landmark cells (PCA plus PLS) assigns each cell to
//! its nearest landmarks. Counting those assignments per donor gives a
//! landmarks x donors matrix; see [sample_diffusion_map] for what happens to it.
//!
//! The phenotype is a sample label, so a PLS fitted on the donors it embeds
//! separates them partly by construction. With `n_folds > 1` the donors are
//! split into folds and each fold is embedded by a model, gene selection
//! included, that never saw it. Distances only make sense within one model, so
//! for every fold the held-out cells, the landmarks and the bandwidths all live
//! in that fold's PLS space; the PCA modality is shared.

use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::Deserialize;
use std::time::Instant;

use crate::multi_modal::wnn::{QueryModality, compute_wnn_query};
use crate::prelude::*;
use crate::sc_analysis::milo_r::count_nhood_cells;
use crate::sc_analysis::scenic::subsample_cells;
use crate::sc_annotation::symphony::generate_knn_cross_with_dist;
use crate::sc_batch_correction::batch_utils::cosine_normalise;
use crate::sc_processing::sketch::{
    LeverageParams, leverage_scores, weighted_sample_without_replacement,
};
use bixverse_core::core::math::pls::kernel_pls;
use manifolds_rs::data::nearest_neighbours::NearestNeighbourParams;
use manifolds_rs::utils::diffusions::PhateTime;
use manifolds_rs::{DiffusionMapsParams, diffusion_maps};

////////////
// Consts //
////////////

/// Cells read per batch when streaming over the store. Every batch holds the
/// full gene axis, so this bounds the transient CSR. Not tuned.
const SLIDE_CELL_BATCH: usize = 8192;

/// Kernel exponent of the cell to landmark WNN. Seurat's `MultiModalNN`
/// default.
const SLIDE_KERNEL_POWER: f32 = 1.0;

/// Floor on a WNN bandwidth, so a cell whose kNN distances are all equal does
/// not divide by zero.
const SLIDE_SIGMA_FLOOR: f32 = 1e-8;

////////////
// Params //
////////////

/// Parameters of [run_scslide].
///
/// Deserialises from a flat dictionary; missing keys take the defaults, which
/// follow Seurat's `PrepareSampleObject` except `n_comp` (the vignette's 20).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ScSlideParams {
    /// Training sketch: cells drawn uniformly per donor
    pub n_cells_per_donor: usize,
    /// Landmark cells drawn from the training sketch by leverage score
    pub n_landmarks: usize,
    /// PLS components
    pub n_comp: usize,
    /// Add the genes most correlated with the phenotype per cell type to the
    /// HVGs (Seurat's `QuickCorTest`)
    pub add_cor_genes: bool,
    /// Fraction of eligible genes kept per cell type and response column
    pub cor_top_prop: f64,
    /// A gene is eligible in a cell type when it is non-zero in this fraction
    /// of its cells
    pub cor_min_pct: f64,
    /// Cell types with fewer training cells are skipped
    pub cor_min_cells: usize,
    /// Upper clip of the scaled training data (Seurat's `scale.max`)
    pub scale_max: f64,
    /// Donor folds for cross-fitting the PLS. `1` fits once on all donors, as
    /// the reference does
    pub n_folds: usize,
    /// Landmark neighbours kept per cell by the WNN
    pub wnn_k: usize,
    /// Landmark candidates per cell and modality
    pub knn_range: usize,
    /// Self kNN size, self included, for the bandwidths: `sigma` is the
    /// distance to the last of these neighbours minus the nearest
    pub sigma_k: usize,
    /// Fixed weights of the PCA and the PLS modality
    pub modality_weights: [f32; 2],
    /// Landmarks each cell is counted against
    pub k_count: usize,
    /// CountSketch rows of the leverage scores
    pub leverage_n_sketch: usize,
    /// JL distortion of the leverage scores
    pub leverage_eps: f64,
    /// kNN parameters of the two self kNN searches. `k` is ignored
    #[serde(flatten)]
    pub knn_params: KnnParams,
}

impl Default for ScSlideParams {
    /// The defaults.
    ///
    /// ### Returns
    ///
    /// The defaults.
    fn default() -> Self {
        let leverage = LeverageParams::default();
        Self {
            n_cells_per_donor: 1000,
            n_landmarks: 2000,
            n_comp: 20,
            add_cor_genes: true,
            cor_top_prop: 0.01,
            cor_min_pct: 0.1,
            cor_min_cells: 100,
            scale_max: 10.0,
            n_folds: 1,
            wnn_k: 20,
            knn_range: 200,
            sigma_k: 20,
            modality_weights: [0.5, 0.5],
            k_count: 5,
            leverage_n_sketch: leverage.n_sketch,
            leverage_eps: leverage.eps,
            knn_params: KnnParams::default(),
        }
    }
}

/// The landmark density side of scSLIDE.
pub struct ScSlideResult {
    /// Landmarks as positions into the input cells
    pub landmarks: Vec<usize>,
    /// Landmark x donor counts, row-major `n_landmarks * n_donors`. Landmark
    /// cells are not counted
    pub counts: Vec<f64>,
    /// Per cell the `k_count` landmarks it was counted against, as positions
    /// into `landmarks`, best first
    pub cell_landmarks: Vec<Vec<usize>>,
    /// The PLS embedding, cells x `n_comp`, each cell from its own fold's
    /// model
    pub pls_embedding: Mat<f32>,
    /// Fold per donor
    pub donor_folds: Vec<usize>,
    /// Training sketch as positions into the input cells
    pub training_cells: Vec<usize>,
    /// Genes the PLS of each fold was fitted on, store indices
    pub pls_genes: Vec<Vec<usize>>,
}

/////////////
// Helpers //
/////////////

/// Uniform per-donor sketch.
///
/// ### Params
///
/// * `donors` - Donor per cell
/// * `n_donors` - Number of donors
/// * `n_per_donor` - Cells to draw per donor; all of a smaller donor
/// * `seed` - Seed, offset by the donor index
///
/// ### Returns
///
/// Cell positions, ascending.
fn training_sketch(
    donors: &[usize],
    n_donors: usize,
    n_per_donor: usize,
    seed: usize,
) -> Vec<usize> {
    let mut by_donor: Vec<Vec<usize>> = vec![Vec::new(); n_donors];
    for (i, &d) in donors.iter().enumerate() {
        by_donor[d].push(i);
    }
    let mut out: Vec<usize> = by_donor
        .iter()
        .enumerate()
        .flat_map(|(d, cells)| subsample_cells(cells, n_per_donor, seed + d))
        .collect();
    out.sort_unstable();
    out
}

/// Assign donors to cross-fitting folds.
///
/// Donors are shuffled, sorted by the argmax of their phenotype row (the class
/// of a one-hot factor) and dealt round-robin, so every fold sees every class
/// it can.
///
/// ### Params
///
/// * `y_donor` - Phenotype per donor, `n_donors x q`
/// * `n_folds` - Number of folds
/// * `seed` - Seed
///
/// ### Returns
///
/// Fold per donor. All zero for `n_folds == 1`.
fn assign_folds(y_donor: MatRef<f64>, n_folds: usize, seed: usize) -> Vec<usize> {
    let n = y_donor.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed as u64));
    let class = |d: usize| {
        (0..y_donor.ncols())
            .max_by(|&a, &b| y_donor[(d, a)].total_cmp(&y_donor[(d, b)]))
            .unwrap_or(0)
    };
    order.sort_by_key(|&d| class(d));
    let mut folds = vec![0; n];
    for (pos, &d) in order.iter().enumerate() {
        folds[d] = pos % n_folds;
    }
    folds
}

/// Per (cell type, fold) sums for the gene-phenotype correlations.
struct CorSums {
    /// Number of responses
    q: usize,
    /// Number of genes
    n_genes: usize,
    /// Cells per key
    n: Vec<f64>,
    /// Response sums per key, `q` each
    sy: Vec<f64>,
    /// Response sums of squares per key, `q` each
    syy: Vec<f64>,
    /// Gene sums per key, `n_genes` each
    sx: Vec<f64>,
    /// Gene sums of squares per key, `n_genes` each
    sxx: Vec<f64>,
    /// Gene times response sums per key, `n_genes * q` each, gene-major
    sxy: Vec<f64>,
    /// Non-zero counts per key, `n_genes` each
    nnz: Vec<f64>,
}

impl CorSums {
    /// Zeroed sums.
    ///
    /// ### Params
    ///
    /// * `n_keys` - Number of (cell type, fold) keys
    /// * `n_genes` - Number of genes
    /// * `q` - Number of responses
    ///
    /// ### Returns
    ///
    /// The sums.
    fn new(n_keys: usize, n_genes: usize, q: usize) -> Self {
        Self {
            q,
            n_genes,
            n: vec![0.0; n_keys],
            sy: vec![0.0; n_keys * q],
            syy: vec![0.0; n_keys * q],
            sx: vec![0.0; n_keys * n_genes],
            sxx: vec![0.0; n_keys * n_genes],
            sxy: vec![0.0; n_keys * n_genes * q],
            nnz: vec![0.0; n_keys * n_genes],
        }
    }

    /// Add `other` into `self`.
    ///
    /// ### Params
    ///
    /// * `other` - Sums over the same keys
    fn merge(&mut self, other: &CorSums) {
        for (a, b) in [
            (&mut self.n, &other.n),
            (&mut self.sy, &other.sy),
            (&mut self.syy, &other.syy),
            (&mut self.sx, &other.sx),
            (&mut self.sxx, &other.sxx),
            (&mut self.sxy, &other.sxy),
            (&mut self.nnz, &other.nnz),
        ] {
            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        }
    }

    /// Sums over a set of keys, as a one-key [CorSums].
    ///
    /// ### Params
    ///
    /// * `keys` - Keys to add up
    ///
    /// ### Returns
    ///
    /// The pooled sums.
    fn pool(&self, keys: &[usize]) -> CorSums {
        let (g, q) = (self.n_genes, self.q);
        let mut out = CorSums::new(1, g, q);
        for &k in keys {
            out.n[0] += self.n[k];
            let add = |dst: &mut [f64], src: &[f64], w: usize| {
                dst.iter_mut()
                    .zip(&src[k * w..(k + 1) * w])
                    .for_each(|(a, b)| *a += b)
            };
            add(&mut out.sy, &self.sy, q);
            add(&mut out.syy, &self.syy, q);
            add(&mut out.sx, &self.sx, g);
            add(&mut out.sxx, &self.sxx, g);
            add(&mut out.sxy, &self.sxy, g * q);
            add(&mut out.nnz, &self.nnz, g);
        }
        out
    }
}

/// Stream the training sketch once and collect the correlation sums.
///
/// ### Params
///
/// * `reader` - Cell-major reader
/// * `rows` - Store indices of the training cells
/// * `keys` - (cell type, fold) key per training cell
/// * `n_keys` - Number of keys
/// * `y_rows` - Phenotype per training cell, `rows.len() x q`
///
/// ### Returns
///
/// The [CorSums].
fn correlation_sums<S: SingleCellReading>(
    reader: &S,
    rows: &[usize],
    keys: &[usize],
    n_keys: usize,
    y_rows: MatRef<f64>,
) -> Result<CorSums, ScErrors> {
    let n_genes = reader.dims().total_genes;
    let q = y_rows.ncols();
    let mut sums = CorSums::new(n_keys, n_genes, q);

    for (b, batch) in rows.chunks(SLIDE_CELL_BATCH).enumerate() {
        let offset = b * SLIDE_CELL_BATCH;
        let csr = reader.read_cells_sparse::<u32>(batch, &DataLayerReturn::Norm)?;
        let norm = csr.data_2.as_ref().expect("norm layer requested");
        let part = (0..batch.len())
            .into_par_iter()
            .fold(
                || CorSums::new(n_keys, n_genes, q),
                |mut acc, i| {
                    let r = offset + i;
                    let k = keys[r];
                    acc.n[k] += 1.0;
                    for c in 0..q {
                        let y = y_rows[(r, c)];
                        acc.sy[k * q + c] += y;
                        acc.syy[k * q + c] += y * y;
                    }
                    for p in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                        let (g, x) = (csr.indices[p] as usize, norm[p] as f64);
                        acc.sx[k * n_genes + g] += x;
                        acc.sxx[k * n_genes + g] += x * x;
                        acc.nnz[k * n_genes + g] += 1.0;
                        let base = (k * n_genes + g) * q;
                        for c in 0..q {
                            acc.sxy[base + c] += x * y_rows[(r, c)];
                        }
                    }
                    acc
                },
            )
            .reduce_with(|mut a, b| {
                a.merge(&b);
                a
            });
        if let Some(part) = part {
            sums.merge(&part);
        }
    }
    Ok(sums)
}

/// Genes most correlated with the phenotype, as Seurat's `QuickCorTest`.
///
/// Per pooled cell type with at least `min_cells` cells and per response
/// column that varies within it, the genes non-zero in at least `min_pct` of
/// its cells are ranked by absolute Pearson correlation and the top
/// `ceil(top_prop * eligible)` kept. The union over cell types and columns is
/// returned.
///
/// ### Params
///
/// * `pooled` - One pooled [CorSums] per cell type
/// * `params` - The [ScSlideParams]
///
/// ### Returns
///
/// Store gene indices, ascending.
fn select_cor_genes(pooled: &[CorSums], params: &ScSlideParams) -> Vec<usize> {
    let mut keep = FxHashSet::default();
    for s in pooled {
        let n = s.n[0];
        if n < params.cor_min_cells as f64 || n < 2.0 {
            continue;
        }
        let eligible: Vec<usize> = (0..s.n_genes)
            .filter(|&g| s.nnz[g] >= params.cor_min_pct * n)
            .collect();
        let n_top = (params.cor_top_prop * eligible.len() as f64).ceil() as usize;
        for c in 0..s.q {
            let my = s.sy[c] / n;
            let vy = s.syy[c] / n - my * my;
            if vy <= 0.0 {
                continue;
            }
            let mut scored: Vec<(f64, usize)> = eligible
                .iter()
                .filter_map(|&g| {
                    let mx = s.sx[g] / n;
                    let vx = s.sxx[g] / n - mx * mx;
                    let cov = s.sxy[g * s.q + c] / n - mx * my;
                    let r = cov / (vx * vy).sqrt();
                    r.is_finite().then_some((r.abs(), g))
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            keep.extend(scored.iter().take(n_top).map(|&(_, g)| g));
        }
    }
    let mut out: Vec<usize> = keep.into_iter().collect();
    out.sort_unstable();
    out
}

/// Read cells restricted to a gene set, as a CSR over those genes.
///
/// ### Params
///
/// * `reader` - Cell-major reader
/// * `rows` - Store indices of the cells
/// * `gene_pos` - Column per store gene, `usize::MAX` for genes left out
/// * `n_cols` - Number of kept genes
///
/// ### Returns
///
/// CSR `rows.len() x n_cols` of the normalised values, in `f64`.
fn read_gene_subset<S: SingleCellReading>(
    reader: &S,
    rows: &[usize],
    gene_pos: &[usize],
    n_cols: usize,
) -> Result<CompressedSparseData2<f64>, ScErrors> {
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for batch in rows.chunks(SLIDE_CELL_BATCH) {
        let csr = reader.read_cells_sparse::<u32>(batch, &DataLayerReturn::Norm)?;
        let norm = csr.data_2.as_ref().expect("norm layer requested");
        for i in 0..batch.len() {
            let mut row: Vec<(u32, f64)> = (csr.indptr[i] as usize..csr.indptr[i + 1] as usize)
                .filter_map(|p| {
                    let c = gene_pos[csr.indices[p] as usize];
                    (c != usize::MAX).then_some((c as u32, norm[p] as f64))
                })
                .collect();
            row.sort_unstable_by_key(|e| e.0);
            for (c, v) in row {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len() as u32);
        }
    }
    Ok(CompressedSparseData2::from_parts(
        data,
        indices,
        indptr,
        None,
        CompressedSparseFormat::Csr,
        (rows.len(), n_cols),
    ))
}

/// Column means and sample standard deviations of a CSR.
///
/// ### Params
///
/// * `x` - CSR matrix
/// * `rows` - Rows to use
///
/// ### Returns
///
/// `(means, sds)`; a zero sd is set to 1 as Seurat's projection does.
fn column_stats(x: &CompressedSparseData2<f64>, rows: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let p = x.shape.1;
    let n = rows.len() as f64;
    let mut s = vec![0.0; p];
    let mut ss = vec![0.0; p];
    for &i in rows {
        for k in x.indptr[i] as usize..x.indptr[i + 1] as usize {
            let (j, v) = (x.indices[k] as usize, x.data[k]);
            s[j] += v;
            ss[j] += v * v;
        }
    }
    let means: Vec<f64> = s.iter().map(|v| v / n).collect();
    let sds = (0..p)
        .map(|j| {
            let var = (ss[j] - n * means[j] * means[j]) / (n - 1.0);
            let sd = var.max(0.0).sqrt();
            if sd > 0.0 { sd } else { 1.0 }
        })
        .collect();
    (means, sds)
}

/// The PLS design of one fold: training rows over the fold's genes, clipped at
/// `mean + scale_max * sd` and divided by `sd`.
///
/// Seurat clips the z-scores from above only, and a value above the mean is
/// always a stored one, so clipping the raw stored values before scaling is
/// exact and keeps the matrix sparse. The centring happens inside the PLS.
///
/// ### Params
///
/// * `x` - Training CSR over the union gene set
/// * `rows` - Training rows of this fold
/// * `cols` - Columns of this fold's genes
/// * `means` - Column means over `rows`, per entry of `cols`
/// * `sds` - Column sds over `rows`, per entry of `cols`
/// * `scale_max` - Upper clip of the z-scores
///
/// ### Returns
///
/// CSR `rows.len() x cols.len()`.
fn scaled_design(
    x: &CompressedSparseData2<f64>,
    rows: &[usize],
    cols: &[usize],
    means: &[f64],
    sds: &[f64],
    scale_max: f64,
) -> CompressedSparseData2<f64> {
    let mut col_pos = vec![usize::MAX; x.shape.1];
    for (k, &c) in cols.iter().enumerate() {
        col_pos[c] = k;
    }
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for &i in rows {
        for k in x.indptr[i] as usize..x.indptr[i + 1] as usize {
            let c = col_pos[x.indices[k] as usize];
            if c == usize::MAX {
                continue;
            }
            let v = x.data[k].min(means[c] + scale_max * sds[c]);
            indices.push(c as u32);
            data.push(v / sds[c]);
        }
        indptr.push(indices.len() as u32);
    }
    CompressedSparseData2::from_parts(
        data,
        indices,
        indptr,
        None,
        CompressedSparseFormat::Csr,
        (rows.len(), cols.len()),
    )
}

/// Project every cell onto a PLS model: `((x - mean) / sd) R`.
///
/// Uses the affine split `(x / sd) R - (mean / sd)' R`, so only stored values
/// cost anything. No clipping, as Seurat's `ProjectCellEmbeddings`.
///
/// ### Params
///
/// * `reader` - Cell-major reader
/// * `cells` - Store indices of all cells, in output order
/// * `genes` - Store genes of the model
/// * `means` - Training means per gene
/// * `sds` - Training sds per gene
/// * `projection` - PLS projection, `genes.len() x n_comp`
///
/// ### Returns
///
/// Scores, cells x `n_comp`.
fn project_all_cells<S: SingleCellReading>(
    reader: &S,
    cells: &[usize],
    genes: &[usize],
    means: &[f64],
    sds: &[f64],
    projection: MatRef<f64>,
) -> Result<Mat<f32>, ScErrors> {
    let a = projection.ncols();
    let mut gene_pos = vec![usize::MAX; reader.dims().total_genes];
    for (k, &g) in genes.iter().enumerate() {
        gene_pos[g] = k;
    }
    // R / sd, row-major, and the constant (mean / sd)' R
    let r_scaled: Vec<f64> = (0..genes.len())
        .flat_map(|j| (0..a).map(move |c| projection[(j, c)] / sds[j]))
        .collect();
    let offset: Vec<f64> = (0..a)
        .map(|c| {
            (0..genes.len())
                .map(|j| means[j] * r_scaled[j * a + c])
                .sum()
        })
        .collect();

    let mut out = Mat::<f32>::zeros(cells.len(), a);
    for (b, batch) in cells.chunks(SLIDE_CELL_BATCH).enumerate() {
        let csr = reader.read_cells_sparse::<u32>(batch, &DataLayerReturn::Norm)?;
        let norm = csr.data_2.as_ref().expect("norm layer requested");
        let rows: Vec<Vec<f64>> = (0..batch.len())
            .into_par_iter()
            .map(|i| {
                let mut z: Vec<f64> = offset.iter().map(|o| -o).collect();
                for p in csr.indptr[i] as usize..csr.indptr[i + 1] as usize {
                    let j = gene_pos[csr.indices[p] as usize];
                    if j == usize::MAX {
                        continue;
                    }
                    let v = norm[p] as f64;
                    z.iter_mut()
                        .zip(&r_scaled[j * a..(j + 1) * a])
                        .for_each(|(z, r)| *z += v * r);
                }
                z
            })
            .collect();
        for (i, z) in rows.iter().enumerate() {
            for (c, &v) in z.iter().enumerate() {
                out[(b * SLIDE_CELL_BATCH + i, c)] = v as f32;
            }
        }
    }
    Ok(out)
}

/// Nearest-neighbour distance and WNN bandwidth per cell from a self kNN.
///
/// `sigma_k` counts self, as Seurat's `FindModalityWeights` does: with self
/// excluded, `nearest` is the first distance and `sigma` the
/// `(sigma_k - 1)`-th minus `nearest`.
///
/// ### Params
///
/// * `emb` - L2-normalised embedding, cells x d
/// * `rows` - Rows that need a bandwidth
/// * `params` - The [ScSlideParams]
/// * `seed` - Seed of the kNN index
/// * `verbose` - Verbosity of the search
///
/// ### Returns
///
/// `(nearest, sigma)` per entry of `rows`.
fn bandwidths(
    emb: MatRef<f32>,
    rows: &[usize],
    params: &ScSlideParams,
    seed: usize,
    verbose: bool,
) -> Result<(Vec<f32>, Vec<f32>), ScErrors> {
    let k = params.sigma_k - 1;
    let mut knn_params = params.knn_params.clone();
    knn_params.k = k;
    let (_, dist) = generate_knn_with_dist(emb, &knn_params, true, false, seed, verbose)?;
    let dist = dist.expect("return_dist = true always yields distances");
    Ok(rows
        .iter()
        .map(|&i| {
            let d = &dist[i];
            let nearest = d.first().copied().unwrap_or(0.0);
            let far = d.get(k - 1).or(d.last()).copied().unwrap_or(0.0);
            (nearest, (far - nearest).max(SLIDE_SIGMA_FLOOR))
        })
        .unzip())
}

/// Landmark candidates per query cell: exact kNN against the landmark rows.
///
/// ### Params
///
/// * `emb` - L2-normalised embedding, cells x d
/// * `landmarks` - Landmark rows
/// * `rows` - Query rows
/// * `k` - Candidates per query row
/// * `seed` - Seed
///
/// ### Returns
///
/// Per query row, positions into `landmarks`, nearest first.
fn landmark_candidates(
    emb: MatRef<f32>,
    landmarks: &[usize],
    rows: &[usize],
    k: usize,
    seed: usize,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let reference =
        Mat::<f32>::from_fn(landmarks.len(), emb.ncols(), |i, j| emb[(landmarks[i], j)]);
    let query = Mat::<f32>::from_fn(rows.len(), emb.ncols(), |i, j| emb[(rows[i], j)]);
    let knn_params = KnnParams {
        knn_method: "exhaustive".into(),
        k,
        ..KnnParams::default()
    };
    let (idx, _) =
        generate_knn_cross_with_dist(reference.as_ref(), query.as_ref(), &knn_params, seed, false)?;
    Ok(idx)
}

/// Rows of a matrix as a new matrix.
///
/// ### Params
///
/// * `m` - The matrix
/// * `rows` - Rows to take
///
/// ### Returns
///
/// `rows.len() x m.ncols()`.
fn take_rows(m: MatRef<f32>, rows: &[usize]) -> Mat<f32> {
    Mat::from_fn(rows.len(), m.ncols(), |i, j| m[(rows[i], j)])
}

/// One modality ready for the landmark WNN.
struct PreparedModality {
    /// L2-normalised query rows
    query: Mat<f32>,
    /// L2-normalised landmark rows
    landmarks: Mat<f32>,
    /// Nearest self-kNN distance per query row
    nearest: Vec<f32>,
    /// Bandwidth per query row
    sigma: Vec<f32>,
    /// Landmark candidates per query row
    candidates: Vec<Vec<usize>>,
}

impl PreparedModality {
    /// The same modality restricted to some of its query rows.
    ///
    /// ### Params
    ///
    /// * `rows` - Positions into the current query rows
    ///
    /// ### Returns
    ///
    /// The restricted [PreparedModality].
    fn subset(&self, rows: &[usize]) -> PreparedModality {
        PreparedModality {
            query: take_rows(self.query.as_ref(), rows),
            landmarks: self.landmarks.clone(),
            nearest: rows.iter().map(|&i| self.nearest[i]).collect(),
            sigma: rows.iter().map(|&i| self.sigma[i]).collect(),
            candidates: rows.iter().map(|&i| self.candidates[i].clone()).collect(),
        }
    }

    /// View as a [QueryModality].
    ///
    /// ### Params
    ///
    /// * `weight` - Fixed modality weight
    ///
    /// ### Returns
    ///
    /// The [QueryModality].
    fn as_query(&self, weight: f32) -> QueryModality<'_> {
        QueryModality {
            query: self.query.as_ref(),
            reference: self.landmarks.as_ref(),
            candidates: &self.candidates,
            nearest: &self.nearest,
            sigma: &self.sigma,
            weight,
        }
    }
}

/// L2-normalise an embedding, then the bandwidths of the query rows from a
/// self kNN over all rows and their landmark candidates.
///
/// ### Params
///
/// * `emb` - Embedding of all cells, cells x d
/// * `landmarks` - Landmark rows
/// * `rows` - Query rows
/// * `params` - The [ScSlideParams]
/// * `seed` - Seed
///
/// ### Returns
///
/// The [PreparedModality].
fn prepare_modality(
    emb: MatRef<f32>,
    landmarks: &[usize],
    rows: &[usize],
    params: &ScSlideParams,
    seed: usize,
) -> Result<PreparedModality, ScErrors> {
    let norm = cosine_normalise(&emb.to_owned());
    let (nearest, sigma) = bandwidths(norm.as_ref(), rows, params, seed, false)?;
    let k = params.knn_range.min(landmarks.len());
    let candidates = landmark_candidates(norm.as_ref(), landmarks, rows, k, seed)?;
    Ok(PreparedModality {
        query: take_rows(norm.as_ref(), rows),
        landmarks: take_rows(norm.as_ref(), landmarks),
        nearest,
        sigma,
        candidates,
    })
}

/// The fixed-weight WNN of the query rows against the landmarks.
///
/// ### Params
///
/// * `one` - The first modality
/// * `two` - The second modality, same query rows
/// * `params` - The [ScSlideParams]
///
/// ### Returns
///
/// Per query row the best `wnn_k` landmarks, as positions into the landmarks.
fn landmark_wnn(
    one: &PreparedModality,
    two: &PreparedModality,
    params: &ScSlideParams,
) -> Result<Vec<Vec<usize>>, ScErrors> {
    let modalities = [
        one.as_query(params.modality_weights[0]),
        two.as_query(params.modality_weights[1]),
    ];
    let (idx, _) = compute_wnn_query(&modalities, params.wnn_k, SLIDE_KERNEL_POWER)?;
    Ok(idx)
}

/// Count each non-landmark cell against its top `k_count` landmarks.
///
/// ### Params
///
/// * `wnn_rows` - Per cell its WNN landmarks, best first
/// * `landmarks` - Landmark cells
/// * `donors` - Donor per cell
/// * `n_donors` - Number of donors
/// * `params` - The [ScSlideParams]
///
/// ### Returns
///
/// `(counts, cell_landmarks)`: row-major landmarks x donors and the kept
/// landmarks per cell.
fn count_landmarks(
    wnn_rows: Vec<Vec<usize>>,
    landmarks: &[usize],
    donors: &[usize],
    n_donors: usize,
    params: &ScSlideParams,
) -> Result<(Vec<f64>, Vec<Vec<usize>>), ScErrors> {
    let is_landmark: FxHashSet<usize> = landmarks.iter().copied().collect();
    let cell_landmarks: Vec<Vec<usize>> = wnn_rows
        .into_iter()
        .map(|r| r.into_iter().take(params.k_count).collect())
        .collect();
    let (rows, cols): (Vec<usize>, Vec<usize>) = cell_landmarks
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_landmark.contains(i))
        .flat_map(|(i, ls)| ls.iter().map(move |&l| (i, l)))
        .unzip();
    let counts = count_nhood_cells(&rows, &cols, donors, landmarks.len(), n_donors)?;
    Ok((counts, cell_landmarks))
}

/// Check the inputs of [run_scslide].
///
/// ### Params
///
/// * `n_cells` - Number of cells
/// * `donors` - Donor per cell
/// * `y_donor` - Phenotype per donor
/// * `cell_types` - Optional cell type per cell
/// * `pca` - PCA embedding
/// * `params` - The [ScSlideParams]
///
/// ### Returns
///
/// `Ok(())` or [CoreErrors::InvalidArgument] naming the first problem.
fn check_inputs(
    n_cells: usize,
    donors: &[usize],
    y_donor: MatRef<f64>,
    cell_types: Option<&[usize]>,
    pca: MatRef<f32>,
    params: &ScSlideParams,
) -> Result<(), ScErrors> {
    let fail = |msg: String| Err(CoreErrors::InvalidArgument(msg).into());
    if donors.len() != n_cells || pca.nrows() != n_cells {
        return fail(format!(
            "scSLIDE: {n_cells} cells, but {} donor labels and {} PCA rows.",
            donors.len(),
            pca.nrows()
        ));
    }
    if cell_types.is_some_and(|ct| ct.len() != n_cells) {
        return fail(format!("scSLIDE: need one cell type per cell ({n_cells})."));
    }
    if let Some(&bad) = donors.iter().find(|&&d| d >= y_donor.nrows()) {
        return fail(format!(
            "scSLIDE: donor {bad} has no phenotype row ({} rows).",
            y_donor.nrows()
        ));
    }
    if params.n_folds == 0 || params.n_folds > y_donor.nrows() {
        return fail(format!(
            "scSLIDE: n_folds must lie in 1..={}, got {}.",
            y_donor.nrows(),
            params.n_folds
        ));
    }
    if params.sigma_k < 2 || params.k_count == 0 || params.k_count > params.wnn_k {
        return fail(format!(
            "scSLIDE: need sigma_k >= 2 and 1 <= k_count <= wnn_k, got {}, {}, {}.",
            params.sigma_k, params.k_count, params.wnn_k
        ));
    }
    Ok(())
}

//////////
// Main //
//////////

/// Landmark x donor densities of scSLIDE.
///
/// 1. A uniform sketch of `n_cells_per_donor` cells per donor.
/// 2. `n_landmarks` landmarks from it by leverage score over the HVGs.
/// 3. Per fold, the HVGs plus the phenotype-correlated genes of the fold's
///    training donors, a kernel PLS on their scaled training cells, and every
///    cell projected onto it.
/// 4. A fixed-weight WNN from each held-out cell to the landmarks over the
///    L2-normalised PCA and PLS, with bandwidths from a self kNN per modality.
/// 5. Each non-landmark cell adds one to each of its top `k_count` landmarks
///    in its donor's column.
///
/// ### Params
///
/// * `reader` - Cell-major reader of the store
/// * `cells` - Store indices of the cells to use
/// * `donors` - Donor per cell, `0..n_donors`
/// * `y_donor` - Phenotype per donor, `n_donors x q`; a factor one-hot
/// * `cell_types` - Optional cell type per cell, `0..n_types`, for the
///   correlated genes; without it all cells are one group
/// * `hvg` - Store indices of the highly variable genes
/// * `pca` - PCA embedding of `cells`, cells x d
/// * `params` - The [ScSlideParams]
/// * `seed` - Seed
/// * `verbose` - Print progress
///
/// ### Returns
///
/// The [ScSlideResult].
///
/// ### References
///
/// Jiang et al., Cell, 2026
#[allow(clippy::too_many_arguments)]
pub fn run_scslide<S: SingleCellReading>(
    reader: &S,
    cells: &[usize],
    donors: &[usize],
    y_donor: MatRef<f64>,
    cell_types: Option<&[usize]>,
    hvg: &[usize],
    pca: MatRef<f32>,
    params: &ScSlideParams,
    seed: usize,
    verbose: bool,
) -> Result<ScSlideResult, ScErrors> {
    let n_cells = cells.len();
    check_inputs(n_cells, donors, y_donor, cell_types, pca, params)?;
    let n_donors = y_donor.nrows();
    let q = y_donor.ncols();
    let start = Instant::now();
    let log = |msg: &str| {
        if verbose {
            println!("scSLIDE: {msg} ({:.2?})", start.elapsed());
        }
    };

    // 1. training sketch and folds
    let training = training_sketch(donors, n_donors, params.n_cells_per_donor, seed);
    let train_store: Vec<usize> = training.iter().map(|&i| cells[i]).collect();
    let donor_folds = assign_folds(y_donor, params.n_folds, seed);
    log(&format!("training sketch of {} cells", training.len()));

    // 2. landmarks by leverage over the HVGs, as Seurat does it
    let n_genes = reader.dims().total_genes;
    let mut hvg_pos = vec![usize::MAX; n_genes];
    for (k, &g) in hvg.iter().enumerate() {
        hvg_pos[g] = k;
    }
    let train_hvg = read_gene_subset(reader, &train_store, &hvg_pos, hvg.len())?;
    let live: Vec<usize> = {
        let mut seen = vec![false; hvg.len()];
        train_hvg
            .indices
            .iter()
            .for_each(|&c| seen[c as usize] = true);
        (0..hvg.len()).filter(|&c| seen[c]).collect()
    };
    let train_live = train_hvg.slice_cols(&live).map_err(CoreErrors::from)?;
    let leverage = leverage_scores(
        &train_live,
        &LeverageParams::new(params.leverage_n_sketch, params.leverage_eps),
        seed,
    )?;
    let landmarks: Vec<usize> =
        weighted_sample_without_replacement(&leverage, params.n_landmarks, seed)
            .into_iter()
            .map(|i| training[i])
            .collect();
    drop((train_hvg, train_live));
    log(&format!("{} landmarks", landmarks.len()));

    // 3a. correlated genes per fold, one streaming pass
    let y_train = Mat::<f64>::from_fn(training.len(), q, |r, c| y_donor[(donors[training[r]], c)]);
    let n_types = cell_types.map_or(1, |ct| ct.iter().max().map_or(1, |m| m + 1));
    let fold_genes: Vec<Vec<usize>> = if params.add_cor_genes {
        let keys: Vec<usize> = training
            .iter()
            .map(|&i| cell_types.map_or(0, |ct| ct[i]) * params.n_folds + donor_folds[donors[i]])
            .collect();
        let sums = correlation_sums(
            reader,
            &train_store,
            &keys,
            n_types * params.n_folds,
            y_train.as_ref(),
        )?;
        (0..params.n_folds)
            .map(|f| {
                let pooled: Vec<CorSums> = (0..n_types)
                    .map(|t| {
                        let ks: Vec<usize> = (0..params.n_folds)
                            .filter(|&g| params.n_folds == 1 || g != f)
                            .map(|g| t * params.n_folds + g)
                            .collect();
                        sums.pool(&ks)
                    })
                    .collect();
                let mut genes: Vec<usize> = hvg.to_vec();
                genes.extend(select_cor_genes(&pooled, params));
                genes.sort_unstable();
                genes.dedup();
                genes
            })
            .collect()
    } else {
        let mut genes = hvg.to_vec();
        genes.sort_unstable();
        genes.dedup();
        vec![genes; params.n_folds]
    };
    log("gene sets selected");

    // 3b. training matrix over the union of the fold gene sets
    let union: Vec<usize> = {
        let mut u: Vec<usize> = fold_genes.iter().flatten().copied().collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let mut union_pos = vec![usize::MAX; n_genes];
    for (k, &g) in union.iter().enumerate() {
        union_pos[g] = k;
    }
    let train_union = read_gene_subset(reader, &train_store, &union_pos, union.len())?;

    // 4a. PCA modality, shared by every fold
    let all_rows: Vec<usize> = (0..n_cells).collect();
    let pca_prep = prepare_modality(pca, &landmarks, &all_rows, params, seed)?;
    log("PCA bandwidths and landmark candidates");

    let mut pls_embedding = Mat::<f32>::zeros(n_cells, params.n_comp);
    let mut wnn_rows: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
    for f in 0..params.n_folds {
        let held_out: Vec<usize> = (0..n_cells)
            .filter(|&i| params.n_folds == 1 || donor_folds[donors[i]] == f)
            .collect();
        let train_rows: Vec<usize> = (0..training.len())
            .filter(|&r| params.n_folds == 1 || donor_folds[donors[training[r]]] != f)
            .collect();

        // 3c. scale, fit and project
        let cols: Vec<usize> = fold_genes[f].iter().map(|&g| union_pos[g]).collect();
        let (means_u, sds_u) = column_stats(&train_union, &train_rows);
        let means: Vec<f64> = cols.iter().map(|&c| means_u[c]).collect();
        let sds: Vec<f64> = cols.iter().map(|&c| sds_u[c]).collect();
        let design = scaled_design(
            &train_union,
            &train_rows,
            &cols,
            &means,
            &sds,
            params.scale_max,
        );
        let y_fit = Mat::<f64>::from_fn(train_rows.len(), q, |r, c| y_train[(train_rows[r], c)]);
        let fit = kernel_pls(&design, y_fit.as_ref(), params.n_comp)?;
        let scores = project_all_cells(
            reader,
            cells,
            &fold_genes[f],
            &means,
            &sds,
            fit.projection.as_ref(),
        )?;
        for &i in &held_out {
            for c in 0..params.n_comp {
                pls_embedding[(i, c)] = scores[(i, c)];
            }
        }

        // 4b. PLS modality in this fold's space, then the WNN
        let pls_prep = prepare_modality(scores.as_ref(), &landmarks, &held_out, params, seed)?;
        let idx = landmark_wnn(&pca_prep.subset(&held_out), &pls_prep, params)?;
        for (&i, row) in held_out.iter().zip(idx) {
            wnn_rows[i] = row;
        }
        log(&format!("fold {} of {}", f + 1, params.n_folds));
    }

    // 5. counts, landmark cells left out
    let (counts, cell_landmarks) = count_landmarks(wnn_rows, &landmarks, donors, n_donors, params)?;
    log("landmark counts");

    Ok(ScSlideResult {
        landmarks,
        counts,
        cell_landmarks,
        pls_embedding,
        donor_folds,
        training_cells: training,
        pls_genes: fold_genes,
    })
}

/// Landmark x donor counts from two embeddings you already have.
///
/// Seurat's `FindmmNN` plus `GenerateSampleObject`: a fixed-weight WNN from
/// every cell to the landmarks over the two L2-normalised embeddings, then each
/// non-landmark cell counted against its top `k_count`. [run_scslide] does
/// this per fold with its own PLS; this one takes a custom embedding (a
/// reference mapping, an integrated space) or a PLS fitted elsewhere.
///
/// ### Params
///
/// * `emb_one` - First modality, cells x d1
/// * `emb_two` - Second modality, cells x d2
/// * `landmarks` - Landmark cells, positions into the rows
/// * `donors` - Donor per cell, `0..n_donors`
/// * `n_donors` - Number of donors
/// * `params` - The [ScSlideParams]; only the WNN and count keys are read
/// * `seed` - Seed
///
/// ### Returns
///
/// `(counts, cell_landmarks)`: row-major landmarks x donors, and per cell the
/// `k_count` landmarks it was counted against as positions into `landmarks`.
pub fn landmark_counts(
    emb_one: MatRef<f32>,
    emb_two: MatRef<f32>,
    landmarks: &[usize],
    donors: &[usize],
    n_donors: usize,
    params: &ScSlideParams,
    seed: usize,
) -> Result<(Vec<f64>, Vec<Vec<usize>>), ScErrors> {
    let n_cells = emb_one.nrows();
    if emb_two.nrows() != n_cells || donors.len() != n_cells {
        return Err(CoreErrors::InvalidArgument(format!(
            "landmark_counts: {n_cells} rows in the first embedding, {} in the second, {} donor labels.",
            emb_two.nrows(),
            donors.len()
        ))
        .into());
    }
    if let Some(&bad) = landmarks.iter().find(|&&l| l >= n_cells) {
        return Err(CoreErrors::InvalidArgument(format!(
            "landmark_counts: landmark {bad} is outside the {n_cells} cells."
        ))
        .into());
    }
    if let Some(&bad) = donors.iter().find(|&&d| d >= n_donors) {
        return Err(CoreErrors::InvalidArgument(format!(
            "landmark_counts: donor {bad} is outside 0..{n_donors}."
        ))
        .into());
    }
    let rows: Vec<usize> = (0..n_cells).collect();
    let one = prepare_modality(emb_one, landmarks, &rows, params, seed)?;
    let two = prepare_modality(emb_two, landmarks, &rows, params, seed)?;
    let idx = landmark_wnn(&one, &two, params)?;
    count_landmarks(idx, landmarks, donors, n_donors, params)
}

//////////////////
// Sample level //
//////////////////

/// Parameters of [sample_diffusion_map].
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SampleDmParams {
    /// Diffusion components to return
    pub n_dcs: usize,
    /// Neighbours per sample in the diffusion kernel; the `k`-th sets the
    /// adaptive bandwidth
    pub k: usize,
    /// Diffusion time: components are scaled by `lambda^t`
    pub t: usize,
    /// Density normalisation exponent, 1 for Laplace-Beltrami
    pub alpha_norm: f64,
}

impl Default for SampleDmParams {
    /// The defaults.
    ///
    /// ### Returns
    ///
    /// The defaults.
    fn default() -> Self {
        Self {
            n_dcs: 10,
            k: 10,
            t: 1,
            alpha_norm: 1.0,
        }
    }
}

/// The sample-level side of scSLIDE.
pub struct SampleEmbedding {
    /// Chi-squared residuals, row-major `n_landmarks * n_samples`
    pub residuals: Vec<f64>,
    /// Diffusion components, samples x `n_dcs`
    pub dcs: Mat<f64>,
}

/// Pearson residuals of a contingency table, Seurat's `NormalizeChiSquared`.
///
/// `(c - e) / sqrt(e)` with `e = row_sum * col_sum / total`; cells with
/// `e = 0` (an empty row or column) are 0. No clipping.
///
/// ### Params
///
/// * `counts` - Row-major `n_rows * n_cols` counts
/// * `n_rows` - Rows (landmarks)
/// * `n_cols` - Columns (samples)
///
/// ### Returns
///
/// The residuals, same layout.
pub fn chi_squared_residuals(counts: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let row_sums: Vec<f64> = counts
        .chunks_exact(n_cols)
        .map(|r| r.iter().sum())
        .collect();
    let mut col_sums = vec![0.0; n_cols];
    for row in counts.chunks_exact(n_cols) {
        col_sums.iter_mut().zip(row).for_each(|(s, v)| *s += v);
    }
    let total: f64 = row_sums.iter().sum();
    (0..n_rows * n_cols)
        .map(|k| {
            let e = row_sums[k / n_cols] * col_sums[k % n_cols] / total;
            if e > 0.0 {
                (counts[k] - e) / e.sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Sample embedding from landmark counts: chi-squared residuals, centred per
/// landmark, then a diffusion map over the samples with cosine distance.
///
/// The reference runs destiny on `1 - Pearson` with a complete graph and
/// scales components by `sqrt(lambda)`; this runs manifolds-rs on cosine
/// distance with an adaptive `k`-neighbour kernel and `lambda^t`, so leading
/// components agree up to sign and scale, not exactly.
///
/// ### Params
///
/// * `counts` - Landmark x sample counts, row-major, as [run_scslide] returns
/// * `n_landmarks` - Landmarks
/// * `n_samples` - Samples
/// * `params` - The [SampleDmParams]
/// * `seed` - Seed of the eigensolver
/// * `verbose` - Verbosity of manifolds-rs
///
/// ### Returns
///
/// The [SampleEmbedding].
///
/// ### References
///
/// Coifman & Lafon, ACHA, 2006; Jiang et al., Cell, 2026
pub fn sample_diffusion_map(
    counts: &[f64],
    n_landmarks: usize,
    n_samples: usize,
    params: &SampleDmParams,
    seed: usize,
    verbose: usize,
) -> Result<SampleEmbedding, ScErrors> {
    if counts.len() != n_landmarks * n_samples {
        return Err(CoreErrors::InvalidArgument(format!(
            "counts has {} entries, expected {n_landmarks} x {n_samples}.",
            counts.len()
        ))
        .into());
    }
    if params.k >= n_samples || params.n_dcs + 1 >= n_samples {
        return Err(CoreErrors::InvalidArgument(format!(
            "{n_samples} samples are too few for k = {} and {} components.",
            params.k, params.n_dcs
        ))
        .into());
    }
    let residuals = chi_squared_residuals(counts, n_landmarks, n_samples);

    // samples x landmarks, each landmark centred across samples
    let mut data = vec![0.0; n_samples * n_landmarks];
    for (l, row) in residuals.chunks_exact(n_samples).enumerate() {
        let mean = row.iter().sum::<f64>() / n_samples as f64;
        for (s, v) in row.iter().enumerate() {
            data[s * n_landmarks + l] = v - mean;
        }
    }

    let dm_params = DiffusionMapsParams {
        n_dim: params.n_dcs,
        k: params.k,
        ann_type: "exhaustive".into(),
        nn_params: NearestNeighbourParams {
            dist_metric: "cosine".into(),
            ..NearestNeighbourParams::default()
        },
        alpha_norm: params.alpha_norm,
        t: PhateTime::Fixed(params.t),
        ..DiffusionMapsParams::default()
    };
    let dcs = diffusion_maps(
        (data.as_slice(), n_samples, n_landmarks),
        None,
        dm_params,
        seed,
        verbose,
    )?;

    Ok(SampleEmbedding {
        residuals,
        dcs: Mat::from_fn(n_samples, params.n_dcs, |s, c| dcs[c][s]),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use bixverse_sc_data::store::chunk::CsrCellChunk;
    use bixverse_sc_data::store::reader::ParallelSparseReader;
    use bixverse_sc_data::store::writer::CellGeneSparseWriter;
    use rand::Rng;
    use tempfile::NamedTempFile;

    /// Donors in the synthetic data; the first half are cases.
    const N_DONORS: usize = 12;
    /// Cells per donor.
    const CELLS_PER_DONOR: usize = 200;
    /// Genes; the first five respond to the condition in cell type 0.
    const N_GENES: usize = 40;
    /// Cell types, each with its own block of eight marker genes.
    const N_TYPES: usize = 3;

    /// Synthetic experiment written to a cell-based store.
    struct Synthetic {
        /// Keeps the store alive
        _store: tempfile::TempPath,
        /// Reader over the store
        reader: ParallelSparseReader,
        /// Donor per cell
        donors: Vec<usize>,
        /// Cell type per cell
        types: Vec<usize>,
        /// One-hot condition per donor
        y: Mat<f64>,
        /// A ten-dimensional random projection of the log counts
        pca: Mat<f32>,
    }

    /// Build the synthetic experiment.
    ///
    /// ### Returns
    ///
    /// The [Synthetic] data.
    fn synthetic() -> Synthetic {
        let mut rng = StdRng::seed_from_u64(9);
        let n_cells = N_DONORS * CELLS_PER_DONOR;
        let store = NamedTempFile::with_suffix(".bin").unwrap().into_temp_path();
        let mut writer =
            CellGeneSparseWriter::new(store.to_str().unwrap(), true, n_cells, N_GENES, 1e4)
                .unwrap();
        let proj: Vec<f32> = (0..N_GENES * 10)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let (mut donors, mut types) = (Vec::new(), Vec::new());
        let mut pca = Mat::<f32>::zeros(n_cells, 10);
        for cell in 0..n_cells {
            let (d, t) = (cell / CELLS_PER_DONOR, rng.random_range(0..N_TYPES));
            let case = d < N_DONORS / 2;
            let (mut raw, mut idx) = (Vec::new(), Vec::new());
            for g in 0..N_GENES {
                let mut rate = 1.0;
                if (10 + 8 * t..18 + 8 * t).contains(&g) {
                    rate = 6.0;
                }
                if case && t == 0 && g < 5 {
                    rate = 8.0;
                }
                let v = (rate * rng.random_range(0.0..2.0f64)).floor() as u32;
                if v > 0 {
                    raw.push(v);
                    idx.push(g as u32);
                    for c in 0..10 {
                        pca[(cell, c)] += (v as f32).ln_1p() * proj[g * 10 + c];
                    }
                }
            }
            writer
                .write_cell_chunk(CsrCellChunk::from_data(&raw, &idx, cell, 1e4, true))
                .unwrap();
            donors.push(d);
            types.push(t);
        }
        writer.finalise().unwrap();
        let y = Mat::<f64>::from_fn(N_DONORS, 2, |d, c| {
            ((d < N_DONORS / 2) == (c == 0)) as u8 as f64
        });
        Synthetic {
            reader: ParallelSparseReader::new(store.to_str().unwrap()).unwrap(),
            _store: store,
            donors,
            types,
            y,
            pca,
        }
    }

    /// Small-data parameters.
    ///
    /// ### Params
    ///
    /// * `n_folds` - Cross-fitting folds
    ///
    /// ### Returns
    ///
    /// The [ScSlideParams].
    fn small_params(n_folds: usize) -> ScSlideParams {
        ScSlideParams {
            n_cells_per_donor: 80,
            n_landmarks: 100,
            n_comp: 3,
            cor_min_cells: 20,
            cor_top_prop: 0.1,
            knn_range: 50,
            leverage_n_sketch: 500,
            n_folds,
            ..ScSlideParams::default()
        }
    }

    /// Run scSLIDE on the synthetic data.
    ///
    /// ### Params
    ///
    /// * `s` - The data
    /// * `params` - Parameters
    /// * `seed` - Seed
    ///
    /// ### Returns
    ///
    /// The [ScSlideResult].
    fn run(s: &Synthetic, params: &ScSlideParams, seed: usize) -> ScSlideResult {
        let cells: Vec<usize> = (0..s.donors.len()).collect();
        let hvg: Vec<usize> = (5..N_GENES).collect();
        run_scslide(
            &s.reader,
            &cells,
            &s.donors,
            s.y.as_ref(),
            Some(&s.types),
            &hvg,
            s.pca.as_ref(),
            params,
            seed,
            false,
        )
        .unwrap()
    }

    /// Area under the curve of DC1 for cases against controls.
    ///
    /// ### Params
    ///
    /// * `dcs` - Diffusion components, samples x n_dcs
    ///
    /// ### Returns
    ///
    /// The fraction of (case, control) pairs with the case higher on DC1.
    fn dc1_auc(dcs: &Mat<f64>) -> f64 {
        let half = N_DONORS / 2;
        let wins = (0..half)
            .flat_map(|a| (half..N_DONORS).map(move |b| (a, b)))
            .filter(|&(a, b)| dcs[(a, 0)] > dcs[(b, 0)])
            .count();
        wins as f64 / (half * half) as f64
    }

    #[test]
    fn test_chi_squared_residuals_known_table() {
        let r = chi_squared_residuals(&[10.0, 20.0, 30.0, 40.0], 2, 2);
        approx::assert_relative_eq!(r[0], (10.0 - 12.0) / 12f64.sqrt(), max_relative = 1e-12);
        approx::assert_relative_eq!(r[3], (40.0 - 42.0) / 42f64.sqrt(), max_relative = 1e-12);
        assert_eq!(
            chi_squared_residuals(&[0.0, 0.0, 1.0, 1.0], 2, 2)[..2],
            [0.0, 0.0]
        );
    }

    #[test]
    fn test_assign_folds_stratifies_classes() {
        let y = Mat::<f64>::from_fn(12, 2, |d, c| ((d < 6) == (c == 0)) as u8 as f64);
        let folds = assign_folds(y.as_ref(), 3, 1);
        for f in 0..3 {
            let cases = (0..6).filter(|&d| folds[d] == f).count();
            let controls = (6..12).filter(|&d| folds[d] == f).count();
            assert_eq!((cases, controls), (2, 2));
        }
    }

    #[test]
    fn test_scslide_counts_every_non_landmark_cell() {
        let s = synthetic();
        let params = small_params(1);
        let res = run(&s, &params, 3);
        let n_cells = s.donors.len();

        assert_eq!(res.landmarks.len(), params.n_landmarks);
        let unique: FxHashSet<usize> = res.landmarks.iter().copied().collect();
        assert_eq!(unique.len(), params.n_landmarks);
        assert!(res.landmarks.iter().all(|l| res.training_cells.contains(l)));
        assert_eq!(res.counts.len(), params.n_landmarks * N_DONORS);
        let total: f64 = res.counts.iter().sum();
        assert_eq!(
            total as usize,
            (n_cells - params.n_landmarks) * params.k_count
        );
        assert!(res.cell_landmarks.iter().all(|r| r.len() == params.k_count));
        // the condition genes are not HVGs, so only the correlation test adds them
        assert!(res.pls_genes[0].iter().any(|&g| g < 5));
    }

    #[test]
    fn test_scslide_reproducible() {
        let s = synthetic();
        let params = small_params(1);
        assert_eq!(run(&s, &params, 5).counts, run(&s, &params, 5).counts);
    }

    #[test]
    fn test_scslide_cross_fitted_runs_per_fold() {
        let s = synthetic();
        let params = small_params(3);
        let res = run(&s, &params, 3);
        assert_eq!(res.pls_genes.len(), 3);
        let total: f64 = res.counts.iter().sum();
        assert_eq!(
            total as usize,
            (s.donors.len() - params.n_landmarks) * params.k_count
        );
        assert!(
            (0..s.donors.len())
                .all(|i| (0..params.n_comp).any(|c| res.pls_embedding[(i, c)] != 0.0))
        );
    }

    #[test]
    fn test_sample_diffusion_map_separates_planted_condition() {
        let s = synthetic();
        let params = small_params(1);
        let res = run(&s, &params, 3);
        let dm = SampleDmParams {
            n_dcs: 3,
            k: 4,
            ..SampleDmParams::default()
        };
        let emb =
            sample_diffusion_map(&res.counts, params.n_landmarks, N_DONORS, &dm, 1, 0).unwrap();
        assert_eq!((emb.dcs.nrows(), emb.dcs.ncols()), (N_DONORS, 3));
        let auc = dc1_auc(&emb.dcs);
        assert!(auc.max(1.0 - auc) >= 0.9, "DC1 AUC {auc}");
    }
}
