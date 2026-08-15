//! Highly variable gene selection and principal component analysis (PCA) for
//! meta cells. Works directly on `CompressedSparseData2` structures.

use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::borrow::Cow;

use crate::core::base::info::parse_bin_strategy_type;
use crate::core::base::loess::LoessRegression;
use crate::core::math::MAX_OVERSAMPLING_SINGLE_CELL;
use crate::core::math::pca_svd::*;
use crate::prelude::*;
use crate::single_cell::sc_processing::hvg::*;
use crate::single_cell::sc_processing::pca::{SingleCellPcaParams, SingleCellPcaRes};
use crate::utils::simd::{sum_squared_dev_widen_simd_f32, sum_widen_simd_f32};

/////////
// HVG //
/////////

/// First-pass statistics for one gene of an in-memory sparse matrix.
///
/// The in-memory twin of [`GeneStats`], which cannot be reused here because it
/// carries the extremes as raw `u32` counts.
#[derive(Clone, Copy, Default)]
struct DenseGeneStats {
    /// Mean over every cell, zeros included.
    mean: f32,
    /// Population variance over every cell, zeros included.
    var: f32,
    /// Largest stored value. Zero for a gene with no counts.
    max: f32,
    /// Smallest value, counting the implicit zeros. Zero unless the gene is
    /// expressed in every cell.
    min: f32,
    /// Number of stored (non-zero) entries.
    nnz: usize,
}

///////////////////
// CSR streaming //
///////////////////

/// Per-gene accumulators for one streaming pass over a CSR matrix.
///
/// Every worker owns one of these, so the private scratch is roughly
/// `n_genes * 40` bytes. At thirty-odd thousand genes that stays inside L2, and
/// a CSR row stores its gene indices in ascending order, so a row is a monotone
/// sweep through the arrays rather than a random walk.
struct GeneAccum {
    /// Sum of the transformed stored values.
    sum: Vec<f64>,
    /// Sum of squares of the transformed stored values.
    sum_sq: Vec<f64>,
    /// Number of stored entries.
    nnz: Vec<u32>,
    /// Smallest transformed stored value. Stays `INFINITY` for a gene with no
    /// stored entries.
    min: Vec<f64>,
    /// Largest transformed stored value. Stays `NEG_INFINITY` for a gene with
    /// no stored entries.
    max: Vec<f64>,
}

impl GeneAccum {
    /// Allocate the accumulators for `n_genes` genes.
    ///
    /// ### Params
    ///
    /// * `n_genes` - Number of genes, i.e. the minor axis of the CSR input
    ///
    /// ### Returns
    ///
    /// The zeroed accumulators, with the extremes at their neutral values.
    fn new(n_genes: usize) -> Self {
        Self {
            sum: vec![0.0; n_genes],
            sum_sq: vec![0.0; n_genes],
            nnz: vec![0; n_genes],
            min: vec![f64::INFINITY; n_genes],
            max: vec![f64::NEG_INFINITY; n_genes],
        }
    }

    /// Fold one stored value into the accumulators of its gene.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index of the stored value
    /// * `value` - The transformed value
    #[inline(always)]
    fn push(&mut self, gene: usize, value: f64) {
        self.sum[gene] += value;
        self.sum_sq[gene] += value * value;
        self.nnz[gene] += 1;
        if value < self.min[gene] {
            self.min[gene] = value;
        }
        if value > self.max[gene] {
            self.max[gene] = value;
        }
    }

    /// Fold another worker's accumulators into these.
    ///
    /// ### Params
    ///
    /// * `other` - The accumulators to absorb
    fn merge(&mut self, other: Self) {
        for gene in 0..self.sum.len() {
            self.sum[gene] += other.sum[gene];
            self.sum_sq[gene] += other.sum_sq[gene];
            self.nnz[gene] += other.nnz[gene];
            if other.min[gene] < self.min[gene] {
                self.min[gene] = other.min[gene];
            }
            if other.max[gene] > self.max[gene] {
                self.max[gene] = other.max[gene];
            }
        }
    }
}

/// Accumulate per-gene moments in a single streaming pass over a CSR matrix.
///
/// The column-wise path needs a CSC matrix and therefore a full transpose of
/// the input, which is a serial random-write scatter over `nnz`-sized buffers
/// and dominates everything downstream. Nothing in the first pass actually
/// needs a gene's values to be contiguous: mean, variance and the two extremes
/// are all scatter-adds keyed on the gene index, so they accumulate straight
/// off the incoming CSR instead.
///
/// ### Params
///
/// * `matrix` - The counts, CSR with shape (cells, genes)
/// * `transform` - Applied to every stored value before it is accumulated
///
/// ### Returns
///
/// The merged [GeneAccum] over every gene.
fn accumulate_genes_from_csr<F>(matrix: &CompressedSparseData2<f32>, transform: F) -> GeneAccum
where
    F: Fn(f32) -> f64 + Sync,
{
    let (n_rows, n_genes) = matrix.shape;

    thread_chunks(n_rows)
        .par_iter()
        .map(|&(row_start, row_end)| {
            let mut acc = GeneAccum::new(n_genes);

            // rows are contiguous in the buffers, so one chunk is one slice
            let lo = matrix.indptr[row_start] as usize;
            let hi = matrix.indptr[row_end] as usize;

            for (&gene, &value) in matrix.indices[lo..hi]
                .iter()
                .zip(matrix.data[lo..hi].iter())
            {
                acc.push(gene as usize, transform(value));
            }

            acc
        })
        .reduce(
            || GeneAccum::new(n_genes),
            |mut a, b| {
                a.merge(b);
                a
            },
        )
}

/// First-pass gene statistics from a CSC matrix.
///
/// Column-wise, so every gene is one contiguous slice and the two sums go
/// through the widening SIMD kernels.
///
/// ### Params
///
/// * `csc` - The counts, CSC with shape (cells, genes)
///
/// ### Returns
///
/// One [DenseGeneStats] per gene.
fn gene_stats_from_csc(csc: &CompressedSparseData2<f32>) -> Vec<DenseGeneStats> {
    let n_cells = csc.shape.0;
    let n_cells_f64 = n_cells as f64;

    csc.indptr
        .par_windows(2)
        .map(|window| {
            let slice = &csc.data[window[0] as usize..window[1] as usize];
            let nnz = slice.len();
            if nnz == 0 {
                return DenseGeneStats::default();
            }

            let mean = sum_widen_simd_f32(slice) / n_cells_f64;
            let ss_nonzero = sum_squared_dev_widen_simd_f32(slice, mean);
            let ss_zeros = (n_cells - nnz) as f64 * mean * mean;

            let (min, max) = slice
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });

            DenseGeneStats {
                mean: mean as f32,
                var: ((ss_nonzero + ss_zeros) / n_cells_f64) as f32,
                max,
                min: if nnz < n_cells { 0.0 } else { min },
                nnz,
            }
        })
        .collect()
}

/// First-pass gene statistics from a CSR matrix, without transposing.
///
/// Variance comes out of `sum_sq / n - mean^2` rather than the two-pass
/// deviation sum the CSC path uses, because a streaming pass does not know a
/// gene's mean until it has seen every row. The relative error of that form is
/// `eps * sum_sq / (n * var)`, which for count data stays around `1e-8` in
/// `f64` even for a high-mean, low-dispersion gene.
/// [`disp_stats_from_sums`] already ships the same form.
///
/// ### Params
///
/// * `csr` - The counts, CSR with shape (cells, genes)
///
/// ### Returns
///
/// One [DenseGeneStats] per gene.
fn gene_stats_from_csr(csr: &CompressedSparseData2<f32>) -> Vec<DenseGeneStats> {
    let n_cells = csr.shape.0;
    let n_cells_f64 = n_cells as f64;

    let acc = accumulate_genes_from_csr(csr, |v| v as f64);

    (0..csr.shape.1)
        .map(|gene| {
            let nnz = acc.nnz[gene] as usize;
            if nnz == 0 {
                return DenseGeneStats::default();
            }

            let mean = acc.sum[gene] / n_cells_f64;
            let var = (acc.sum_sq[gene] / n_cells_f64 - mean * mean).max(0.0);

            DenseGeneStats {
                mean: mean as f32,
                var: var as f32,
                max: acc.max[gene] as f32,
                min: if nnz < n_cells {
                    0.0
                } else {
                    acc.min[gene] as f32
                },
                nnz,
            }
        })
        .collect()
}

/// Standardised variance of a gene from its clipped standardised sums.
///
/// Folds the implicit zeros in, then returns the population variance of the
/// clipped standardised values.
///
/// ### Params
///
/// * `sum_std` - Sum of the clipped standardised stored values
/// * `sum_sq_std` - Sum of their squares
/// * `mean` - Mean of the gene over every cell, zeros included
/// * `expected_sd` - Square root of the loess-fitted expected variance
/// * `clip_max` - Symmetric clip applied to the standardised values
/// * `n_zeros` - Number of implicit zeros
/// * `n_cells` - Total number of cells
///
/// ### Returns
///
/// The standardised variance.
#[inline]
fn var_std_from_sums(
    mut sum_std: f64,
    mut sum_sq_std: f64,
    mean: f32,
    expected_sd: f32,
    clip_max: f32,
    n_zeros: usize,
    n_cells: usize,
) -> f32 {
    if n_zeros > 0 {
        let std_zero = ((-mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
        sum_std += n_zeros as f64 * std_zero;
        sum_sq_std += n_zeros as f64 * std_zero * std_zero;
    }

    let n_cells_f64 = n_cells as f64;
    let std_mean = sum_std / n_cells_f64;

    ((sum_sq_std / n_cells_f64) - std_mean * std_mean) as f32
}

/// Second-pass standardised variance from a CSC matrix.
///
/// ### Params
///
/// * `csc` - The counts, CSC with shape (cells, genes)
/// * `stats` - First-pass statistics, one per gene
/// * `expected_var` - Loess-fitted expected variance, one per gene
/// * `clip_max` - Symmetric clip applied to the standardised values
///
/// ### Returns
///
/// The standardised variance per gene.
fn var_std_from_csc(
    csc: &CompressedSparseData2<f32>,
    stats: &[DenseGeneStats],
    expected_var: &[f32],
    clip_max: f32,
) -> Vec<f32> {
    let n_cells = csc.shape.0;

    csc.indptr
        .par_windows(2)
        .zip(stats.par_iter())
        .zip(expected_var.par_iter())
        .map(|((window, stats), &expected_var)| {
            if !clip_is_reachable(stats.mean, stats.min, stats.max, expected_var, clip_max) {
                return stats.var / expected_var;
            }

            let slice = &csc.data[window[0] as usize..window[1] as usize];
            let expected_sd = expected_var.sqrt();
            let mean = stats.mean;

            // `min`/`max` rather than `clamp`: the former discards NaN, the
            // latter propagates it, and a NaN in `var_std` panics the ranking
            // in `select_hvg`.
            let (sum_std, sum_sq_std) = slice.iter().fold((0f64, 0f64), |(s, sq), &val| {
                let norm = ((val - mean) / expected_sd).min(clip_max).max(-clip_max) as f64;
                (s + norm, sq + norm * norm)
            });

            var_std_from_sums(
                sum_std,
                sum_sq_std,
                mean,
                expected_sd,
                clip_max,
                n_cells - slice.len(),
                n_cells,
            )
        })
        .collect()
}

/// Second-pass standardised variance from a CSR matrix, without transposing.
///
/// Genes the clip cannot reach take the closed form `var / expected_var`,
/// because unclipped standardised values sum to zero by construction. When no
/// gene is flagged the streaming pass is skipped outright, which is the common
/// case for metacells: the default clip is `sqrt(n_metacells)`.
///
/// ### Params
///
/// * `csr` - The counts, CSR with shape (cells, genes)
/// * `stats` - First-pass statistics, one per gene
/// * `expected_var` - Loess-fitted expected variance, one per gene
/// * `clip_max` - Symmetric clip applied to the standardised values
///
/// ### Returns
///
/// The standardised variance per gene.
fn var_std_from_csr(
    csr: &CompressedSparseData2<f32>,
    stats: &[DenseGeneStats],
    expected_var: &[f32],
    clip_max: f32,
) -> Vec<f32> {
    let (n_cells, n_genes) = csr.shape;

    let needs_exact: Vec<bool> = stats
        .iter()
        .zip(expected_var.iter())
        .map(|(s, &ev)| clip_is_reachable(s.mean, s.min, s.max, ev, clip_max))
        .collect();

    let mut var_std: Vec<f32> = stats
        .iter()
        .zip(expected_var.iter())
        .map(|(s, &ev)| s.var / ev)
        .collect();

    if !needs_exact.iter().any(|&flagged| flagged) {
        return var_std;
    }

    let expected_sd: Vec<f32> = expected_var.iter().map(|&ev| ev.sqrt()).collect();

    let (sum_std, sum_sq_std) = thread_chunks(n_cells)
        .par_iter()
        .map(|&(row_start, row_end)| {
            let mut sums = vec![0f64; n_genes];
            let mut sums_sq = vec![0f64; n_genes];

            let lo = csr.indptr[row_start] as usize;
            let hi = csr.indptr[row_end] as usize;

            for (&gene, &value) in csr.indices[lo..hi].iter().zip(csr.data[lo..hi].iter()) {
                let gene = gene as usize;
                if !needs_exact[gene] {
                    continue;
                }
                let norm = ((value - stats[gene].mean) / expected_sd[gene])
                    .min(clip_max)
                    .max(-clip_max) as f64;
                sums[gene] += norm;
                sums_sq[gene] += norm * norm;
            }

            (sums, sums_sq)
        })
        .reduce(
            || (vec![0f64; n_genes], vec![0f64; n_genes]),
            |mut a, b| {
                for gene in 0..n_genes {
                    a.0[gene] += b.0[gene];
                    a.1[gene] += b.1[gene];
                }
                a
            },
        );

    for gene in 0..n_genes {
        if !needs_exact[gene] {
            continue;
        }
        var_std[gene] = var_std_from_sums(
            sum_std[gene],
            sum_sq_std[gene],
            stats[gene].mean,
            expected_sd[gene],
            clip_max,
            n_cells - stats[gene].nnz,
            n_cells,
        );
    }

    var_std
}

/// HVG selection via VST from an in-memory sparse matrix.
///
/// Expects raw counts in `data` (no second layer needed). Accepts CSR or CSC
/// input. Neither orientation is transposed: CSC iterates gene slices
/// column-wise, CSR streams the stored values into per-gene accumulators. The
/// transpose used to sit here and dominated the run time, since it is a serial
/// random-write scatter over `nnz`-sized buffers while everything else is
/// gene-parallel.
///
/// Shape must be (cells, genes).
///
/// Both passes accumulate in `f64`: a metacell column sums the counts of dozens
/// of cells, so an `f32` accumulator walks past `2^24` and starts dropping
/// entries. The second pass only looks at a gene when the clip can actually
/// reach one of its values, exactly as [`run_hvg_vst`] does; everywhere else
/// the standardised variance is `var / expected_var` in closed form, because
/// unclipped standardised values sum to zero by construction.
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `loess_span` - The span parameter for the Loess function
/// * `clip_max` - Optional clipping for the data
///
/// ### Returns
///
/// The [HvgRes]
pub fn get_hvg_vst_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    loess_span: f32,
    clip_max: Option<f32>,
) -> HvgRes {
    let n_cells = matrix.shape.0;
    let clip_max = clip_max.unwrap_or((n_cells as f32).sqrt());

    // pass 1: mean, variance and the two extremes per gene
    let stats = match matrix.cs_type {
        CompressedSparseFormat::Csc => gene_stats_from_csc(matrix),
        CompressedSparseFormat::Csr => gene_stats_from_csr(matrix),
    };

    // loess fit on log10 scale
    let (means_log10, vars_log10): (Vec<f32>, Vec<f32>) = stats
        .par_iter()
        .map(|s| (s.mean.log10(), s.var.log10()))
        .unzip();

    let loess = LoessRegression::new(loess_span, 2);
    let loess_res = loess.fit(&means_log10, &vars_log10);

    let expected_var: Vec<f32> = loess_res
        .fitted_vals
        .iter()
        .map(|&fitted| 10_f32.powf(fitted))
        .collect();

    // pass 2: standardised variance, closed form wherever the clip cannot reach
    let var_standardised = match matrix.cs_type {
        CompressedSparseFormat::Csc => var_std_from_csc(matrix, &stats, &expected_var, clip_max),
        CompressedSparseFormat::Csr => var_std_from_csr(matrix, &stats, &expected_var, clip_max),
    };

    HvgRes {
        mean: stats.iter().map(|s| s.mean as f64).collect(),
        var: stats.iter().map(|s| s.var as f64).collect(),
        var_exp: loess_res.fitted_vals.r_float_convert(),
        var_std: var_standardised.r_float_convert(),
    }
}

/// Dispersion-based HVG selection from an in-memory sparse matrix.
///
/// Expects log-normalised data in `data` (i.e. `log1p(x/sum * size_factor)`).
/// Back-transforms via `expm1` internally to match Seurat's
/// `FastExpMean`/`FastLogVMR`.
///
/// Accepts either orientation without transposing, for the reasons given on
/// [`get_hvg_vst_from_sparse`].
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `binning` - The binning strategy to use. One of `"equal_width"` or
///   `"equal_freq"`.
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The [HvgDispersionRes]
pub fn get_hvg_dispersion_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    binning: &str,
    n_bins: usize,
) -> HvgDispersionRes {
    let n_cells = matrix.shape.0;
    let binning = parse_bin_strategy_type(binning).unwrap_or_default();

    // zero entries contribute expm1(0) = 0 so nonzeros suffice
    let (means, dispersions): (Vec<f32>, Vec<f32>) = match matrix.cs_type {
        CompressedSparseFormat::Csc => matrix
            .indptr
            .par_windows(2)
            .map(|window| {
                let slice = &matrix.data[window[0] as usize..window[1] as usize];

                let (sum, sum_sq) = slice.iter().fold((0f64, 0f64), |(s, sq), &val| {
                    let v = (val as f64).exp_m1();
                    (s + v, sq + v * v)
                });

                disp_stats_from_sums(sum, sum_sq, n_cells)
            })
            .unzip(),
        CompressedSparseFormat::Csr => {
            let acc = accumulate_genes_from_csr(matrix, |v| (v as f64).exp_m1());

            (0..matrix.shape.1)
                .map(|gene| disp_stats_from_sums(acc.sum[gene], acc.sum_sq[gene], n_cells))
                .unzip()
        }
    };

    build_disp_result(means, dispersions, binning, n_bins)
}

/// MVB HVG selection from an in-memory sparse matrix.
///
/// Computationally identical to `get_hvg_dispersion_from_sparse`; selection
/// logic (mean/dispersion cutoffs vs. top-N by dispersion) lives on the R side.
///
/// ### Params
///
/// * `matrix` - The count data of the metacell
/// * `binning` - The binning strategy to use. One of `"equal_width"` or
///   `"equal_freq"`.
/// * `n_bins` - Number of bins
///
/// ### Returns
///
/// The [HvgDispersionRes]
pub fn get_hvg_mvb_from_sparse(
    matrix: &CompressedSparseData2<f32>,
    binning: &str,
    n_bins: usize,
) -> HvgDispersionRes {
    get_hvg_dispersion_from_sparse(matrix, binning, n_bins)
}

/////////
// PCA //
/////////

/// PCA on pre-selected HVGs from an in-memory sparse matrix.
///
/// Reads normalised counts from the `data_2` layer (raw counts in `data`
/// are ignored). Densifies, optionally applies the shifted CLR
/// transformation, scales according to `params_pca`, then runs SVD.
/// Shape must be (cells, genes). Uses f64 internally for numerical
/// stability during SVD.
///
/// When `params_pca.clr` is true, `clr_offsets` must be provided and must
/// have been computed against the full gene panel (not the HVG subset),
/// since the row-mean of `log1p(u_ij)` cannot be reconstructed from a
/// gene subset alone.
///
/// ### Params
///
/// * `matrix` - The sparse counts. The `data_2` layer must be populated
///   with `log1p(u * sf)` normalised values.
/// * `no_pcs` - Number of PCs to return.
/// * `params_pca` - PCA parameters, see [SingleCellPcaParams].
/// * `clr_offsets` - Per-cell CLR offsets, required if `params_pca.clr`
///   is true. Length must equal the number of cells.
/// * `seed` - Random seed for the randomised SVD.
///
/// ### Returns
///
/// Tuple of (PCA scores, PCA loadings, singular values).
pub fn pca_on_metacells<T: BixverseNumeric>(
    matrix: &CompressedSparseData2<T, f32>,
    no_pcs: usize,
    params_pca: &SingleCellPcaParams,
    clr_offsets: Option<&[f64]>,
    seed: usize,
) -> SingleCellPcaRes {
    let (n_cells, n_genes) = matrix.shape;

    if params_pca.clr
        && let Some(offs) = clr_offsets
        && offs.len() != n_cells
    {
        return Err(BixverseErrors::OffsetsLengthDoesNotMatchNCells {
            len_offset: offs.len(),
            n_cells,
        });
    }

    // only `data_2` is read below, so the transpose skips the counts layer
    let csc = match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform_single_layer(true)?),
    };
    let vals = csc
        .data_2
        .as_ref()
        .expect("pca_on_metacells requires normalised counts in data_2");

    let vals_active: Cow<[f32]> = if params_pca.clr {
        let sf_inv = 1.0_f32 / params_pca.size_factor;
        let transformed: Vec<f32> = vals
            .par_iter()
            .map(|&v| (v.exp_m1() * sf_inv).ln_1p())
            .collect();
        Cow::Owned(transformed)
    } else {
        Cow::Borrowed(vals.as_slice())
    };

    let row_offsets = if params_pca.clr { clr_offsets } else { None };

    // column major, so one chunk is one gene and the scatter needs no scratch
    let mut buffer = vec![0.0_f64; n_cells * n_genes];

    buffer
        .par_chunks_mut(n_cells)
        .zip(csc.indptr.par_windows(2))
        .for_each(|(col, window)| {
            for idx in window[0] as usize..window[1] as usize {
                col[csc.indices[idx] as usize] = vals_active[idx] as f64;
            }
            if let Some(off) = row_offsets {
                for (v, o) in col.iter_mut().zip(off.iter()) {
                    *v -= o;
                }
            }

            let need_mean = params_pca.mean_center || params_pca.normalise_variance;
            let mean = if need_mean {
                col.iter().sum::<f64>() / n_cells as f64
            } else {
                0.0
            };
            let std_dev = if params_pca.normalise_variance {
                let var: f64 = col
                    .iter()
                    .map(|&x| {
                        let d = x - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / (n_cells as f64 - 1.0);
                var.max(0.0).sqrt()
            } else {
                1.0
            };

            for v in col.iter_mut() {
                if params_pca.mean_center {
                    *v -= mean;
                }
                if params_pca.normalise_variance {
                    *v = if std_dev < 1e-8 { 0.0 } else { *v / std_dev };
                }
            }
        });

    let scaled = MatRef::<f64>::from_column_major_slice(&buffer, n_cells, n_genes);

    let (scores, loadings, s) = if params_pca.randomised {
        let res: RandomSvdResults<f64> = randomised_svd(
            scaled,
            no_pcs,
            seed,
            Some(MAX_OVERSAMPLING_SINGLE_CELL),
            None,
        )?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.v[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| (res.u[(i, j)] * res.s[j]) as f32);
        let s: Vec<f32> = res.s[..no_pcs].iter().map(|&x| x as f32).collect();
        (scores, loadings, s)
    } else {
        let res = scaled
            .thin_svd()
            .map_err(|e| BixverseErrors::FaerSvdError(format!("{e:?}")))?;
        let loadings = Mat::<f32>::from_fn(n_genes, no_pcs, |i, j| res.V()[(i, j)] as f32);
        let scores = Mat::<f32>::from_fn(n_cells, no_pcs, |i, j| {
            (res.U()[(i, j)] * res.S().column_vector()[j]) as f32
        });
        let s: Vec<f32> = res
            .S()
            .column_vector()
            .iter()
            .take(no_pcs)
            .map(|&x| x as f32)
            .collect();
        (scores, loadings, s)
    };

    Ok((scores, loadings, s))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Six cells, four genes. Gene 1 and 3 are expressed everywhere, gene 2 in
    /// a single cell, so the implicit-zero handling and the `min` semantics
    /// both get exercised.
    fn toy_csc() -> CompressedSparseData2<f32> {
        let data: Vec<f32> = vec![
            3.0, 5.0, 2.0, // gene 0
            1.0, 2.0, 1.0, 4.0, 1.0, 3.0, // gene 1
            7.0, // gene 2
            10.0, 12.0, 9.0, 11.0, 10.0, 13.0, // gene 3
        ];
        let indices: Vec<u32> = vec![0, 2, 4, 0, 1, 2, 3, 4, 5, 5, 0, 1, 2, 3, 4, 5];
        let indptr: Vec<u32> = vec![0, 3, 9, 10, 16];

        CompressedSparseData2::new_csc(&data, &indices, &indptr, None::<&[f32]>, (6, 4))
    }

    /// Standardised variance the slow way: densify the gene, standardise every
    /// cell, clip, then take the population variance.
    fn dense_var_std(
        matrix: &CompressedSparseData2<f32>,
        gene: usize,
        res: &HvgRes,
        clip: f32,
    ) -> f64 {
        let n_cells = matrix.shape.0;
        let mut dense = vec![0f32; n_cells];
        for idx in matrix.indptr[gene] as usize..matrix.indptr[gene + 1] as usize {
            dense[matrix.indices[idx] as usize] = matrix.data[idx];
        }

        let mean = res.mean[gene] as f32;
        let expected_sd = (10f64.powf(res.var_exp[gene]) as f32).sqrt();

        let (sum, sum_sq) = dense.iter().fold((0f64, 0f64), |(s, sq), &v| {
            let norm = ((v - mean) / expected_sd).min(clip).max(-clip) as f64;
            (s + norm, sq + norm * norm)
        });

        let n = n_cells as f64;
        (sum_sq / n) - (sum / n) * (sum / n)
    }

    /// Both branches of the second pass have to land on the dense reference:
    /// the closed form when the clip cannot reach any value, and the scan when
    /// it clips every single one.
    #[test]
    fn test_vst_clip_branches_match_the_dense_reference() {
        let matrix = toy_csc();

        for clip in [1e6_f32, 0.25] {
            let res = get_hvg_vst_from_sparse(&matrix, 0.3, Some(clip));

            for gene in 0..matrix.shape.1 {
                assert_relative_eq!(
                    res.var_std[gene],
                    dense_var_std(&matrix, gene, &res, clip),
                    epsilon = 1e-4
                );
            }
        }
    }

    /// The CSR path never transposes, so it has to reproduce the column-wise
    /// path exactly. Both clip settings run, which covers the closed form and
    /// the streaming exact pass.
    #[test]
    fn test_vst_csr_matches_csc() {
        let csc = toy_csc();
        let csr = csc.transform();
        assert!(csr.cs_type.is_csr());

        for clip in [1e6_f32, 0.25] {
            let from_csc = get_hvg_vst_from_sparse(&csc, 0.3, Some(clip));
            let from_csr = get_hvg_vst_from_sparse(&csr, 0.3, Some(clip));

            for gene in 0..csc.shape.1 {
                assert_relative_eq!(from_csr.mean[gene], from_csc.mean[gene], epsilon = 1e-5);
                assert_relative_eq!(from_csr.var[gene], from_csc.var[gene], epsilon = 1e-5);
                assert_relative_eq!(
                    from_csr.var_exp[gene],
                    from_csc.var_exp[gene],
                    epsilon = 1e-5
                );
                assert_relative_eq!(
                    from_csr.var_std[gene],
                    from_csc.var_std[gene],
                    epsilon = 1e-5
                );
            }
        }
    }

    /// Same agreement check for the dispersion path.
    #[test]
    fn test_dispersion_csr_matches_csc() {
        let csc = toy_csc();
        let csr = csc.transform();

        let from_csc = get_hvg_dispersion_from_sparse(&csc, "equal_width", 2);
        let from_csr = get_hvg_dispersion_from_sparse(&csr, "equal_width", 2);

        for gene in 0..csc.shape.1 {
            assert_relative_eq!(from_csr.mean[gene], from_csc.mean[gene], epsilon = 1e-5);
            assert_relative_eq!(
                from_csr.dispersion[gene],
                from_csc.dispersion[gene],
                epsilon = 1e-5
            );
            assert_eq!(from_csr.bin[gene], from_csc.bin[gene]);
        }
    }

    /// `ExpMean` and `LogVMR` against the Seurat definitions, worked out on the
    /// densified column.
    #[test]
    fn test_dispersion_matches_the_seurat_definition() {
        let matrix = toy_csc();
        let res = get_hvg_dispersion_from_sparse(&matrix, "equal_width", 2);

        let n_cells = matrix.shape.0;
        for gene in 0..matrix.shape.1 {
            let mut dense = vec![0f64; n_cells];
            for idx in matrix.indptr[gene] as usize..matrix.indptr[gene + 1] as usize {
                dense[matrix.indices[idx] as usize] = matrix.data[idx] as f64;
            }

            let values: Vec<f64> = dense.iter().map(|v| v.exp_m1()).collect();
            let n = n_cells as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);

            assert_relative_eq!(res.mean[gene], mean.ln_1p(), epsilon = 1e-4);
            assert_relative_eq!(res.dispersion[gene], (var / mean).ln(), epsilon = 1e-4);
        }
    }
}
