//! Contains helpers for the generation of synthetic data for testing
//! algorithms. Has the option to create single cell-like data with defined
//! cell types, optional batch effects and different cell abundance
//! distributions per given sample, plus a DIALOGUE fixture carrying a planted
//! multicellular programme.

use std::f64;

use faer::Mat;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use rustc_hash::FxHashMap;

use crate::prelude::*;
use crate::single_cell::sc_processing::cellsweep::MIN_EMPTY_DROPLETS;

////////////////////////////////
// Synthetic single cell data //
////////////////////////////////

///////////
// Enums //
///////////

/// Enum defining the strength of the batch effect in the synthetic data
#[derive(Clone, Copy, Debug)]
pub enum BatchEffectStrength {
    /// Weak batch effects
    Weak,
    /// Medium batch effects
    Medium,
    /// Strong batch effecst
    Strong,
}

/// Helper function to get the Batch effect strength
///
/// ### Params
///
/// * `s` - Type of KNN algorithm to use
///
/// ### Returns
///
/// Option of the BatchEffectStrength
pub fn parse_batch_effect_strength(s: &str) -> Option<BatchEffectStrength> {
    match s.to_lowercase().as_str() {
        "weak" => Some(BatchEffectStrength::Weak),
        "medium" => Some(BatchEffectStrength::Medium),
        "strong" => Some(BatchEffectStrength::Strong),
        _ => None,
    }
}

/// Enum defining the sample bias, i.e., how many cells of one type are over (or
/// underrepresented) in a given sample
#[derive(Clone, Copy, Debug)]
pub enum SampleBias {
    /// Even distribution of cell types across samples
    Even,
    /// Slightly uneven distribution
    SlightlyUneven,
    /// Very uneven distribution with strong bias
    VeryUneven,
}

/// Helper function to get the Batch effect strength
///
/// ### Params
///
/// * `s` - Type of sample bias to use
///
/// ### Returns
///
/// Option of the SampleBias
pub fn parse_sample_bias(s: &str) -> Option<SampleBias> {
    match s.to_lowercase().as_str() {
        "even" => Some(SampleBias::Even),
        "slightly_uneven" => Some(SampleBias::SlightlyUneven),
        "very_uneven" => Some(SampleBias::VeryUneven),
        _ => None,
    }
}

/// Structure to keep the CellTypeConfig
#[derive(Clone, Debug)]
pub struct CellTypeConfig {
    /// Indices are the marker genes for this specific cell type
    pub marker_genes: Vec<usize>,
}

/////////////
// Helpers //
/////////////

/// Helper function to sample from a Poisson distribution
///
/// Uses Knuth's algorithm for lambda < 30 and transformed rejection method for
/// lambda >= 30.
///
/// ### Params
///
/// * `rng` - Random number generator
/// * `lambda` - Rate parameter
///
/// ### Returns
///
/// A sample from Poisson(λ)
fn poisson_sample<R: Rng>(rng: &mut R, lambda: f64) -> u32 {
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.random::<f64>();
            if p <= l {
                return (k - 1) as u32;
            }
        }
    } else {
        let beta = std::f64::consts::PI / (3.0 * lambda).sqrt();
        let alpha = beta * lambda;
        let k = (2.83 + 5.1 / lambda).ln();

        loop {
            let u = rng.random::<f64>();
            let x = (alpha - ((1.0 - u) / u).ln()) / beta;
            let n = (x + 0.5).floor();
            if n < 0.0 {
                continue;
            }

            let v = rng.random::<f64>();
            let y = alpha - beta * x;
            let lhs = y + (v / (1.0 + y.exp()).powi(2)).ln();
            let rhs = k + n * lambda.ln() - (1..=(n as u32)).map(|i| (i as f64).ln()).sum::<f64>();

            if lhs <= rhs {
                return n as u32;
            }
        }
    }
}

/// Turns a probability vector into a cumulative one for sampling
///
/// The last entry is forced to exactly one, so a uniform draw always lands
/// somewhere even after the accumulated rounding error.
///
/// ### Params
///
/// * `weights` - Probabilities, summing to one
///
/// ### Returns
///
/// The cumulative distribution
fn to_cdf(weights: &[f64]) -> Vec<f64> {
    let mut cum = 0.0;
    let mut cdf: Vec<f64> = weights
        .iter()
        .map(|w| {
            cum += w;
            cum
        })
        .collect();
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

/// Draws one categorical index from a cumulative distribution
///
/// ### Params
///
/// * `rng` - Random number generator
/// * `cdf` - Cumulative distribution, as built by [to_cdf]
///
/// ### Returns
///
/// The drawn index
fn draw_from_cdf<R: Rng>(rng: &mut R, cdf: &[f64]) -> usize {
    let u: f64 = rng.random();
    cdf.partition_point(|&c| c < u).min(cdf.len() - 1)
}

////////////////////
// Main functions //
////////////////////

/// Helper function to create synthetic data with specific cell types
///
/// ### Params
///
/// * `nrow` - Number of rows (cells).
/// * `ncol` - Number of columns (genes).
/// * `cell_type_configs` - A vector of cell type configurations.
/// * `n_batches` - Number of batches to introduce in the data.
/// * `batch_effect_strength` - String indicating the strength of the batch
///   effect to add.
/// * `seed` - Integer for reproducibility purposes
///
/// ### Returns
///
/// A tuple with `(csr data, indices of cell types)`
pub fn create_celltype_sparse_csr_data(
    nrow: usize,
    ncol: usize,
    cell_type_configs: Vec<CellTypeConfig>,
    n_batches: usize,
    batch_effect_strength: &str,
    seed: usize,
) -> (CompressedSparseData2<u32>, Vec<usize>, Vec<usize>) {
    let batch_strength =
        parse_batch_effect_strength(batch_effect_strength).unwrap_or(BatchEffectStrength::Strong);

    let mut indptr = Vec::with_capacity(nrow + 1);
    let mut indices = Vec::with_capacity(nrow * 100);
    let mut data = Vec::with_capacity(nrow * 100);
    let mut cell_type_labels = Vec::with_capacity(nrow);
    let mut batch_labels = Vec::with_capacity(nrow);
    indptr.push(0);

    let n_cell_types = cell_type_configs.len();
    let mut temp_vec = Vec::with_capacity(ncol);

    let mut gene_rng = StdRng::seed_from_u64(seed as u64);
    let mut gene_base_mean = vec![0.0; ncol];
    let mut gene_dispersion = vec![0.0; ncol];

    // Batch effect parameters based on strength
    let (base_range, max_range, systematic_mult, module_mult) = match batch_strength {
        BatchEffectStrength::Weak => (0.8, 1.5, 0.3, 1.3),
        BatchEffectStrength::Medium => (0.5, 3.0, 1.5, 2.5),
        BatchEffectStrength::Strong => (0.3, 5.0, 4.0, 4.0),
    };

    let mut batch_effect = vec![vec![1.0; ncol]; n_batches];

    for batch_idx in 1..n_batches {
        for gene_idx in 0..ncol {
            let u: f64 = gene_rng.random();
            if gene_idx % 5 != 0 {
                batch_effect[batch_idx][gene_idx] = base_range + u * max_range;
            } else {
                batch_effect[batch_idx][gene_idx] = base_range * 2.0 + u * (max_range / 2.0);
            }
        }
    }

    let mut marker_to_celltype = FxHashMap::default();
    for (ct_idx, config) in cell_type_configs.iter().enumerate() {
        for &gene_idx in &config.marker_genes {
            marker_to_celltype.insert(gene_idx, ct_idx);
        }
    }

    for gene_idx in 0..ncol {
        let u: f64 = gene_rng.random();

        if marker_to_celltype.contains_key(&gene_idx) {
            gene_base_mean[gene_idx] = 3.0 + u * 8.0;
            gene_dispersion[gene_idx] = 0.5 + u * 1.5;
        } else {
            let exp = (-u * 3.5).exp();
            gene_base_mean[gene_idx] = 0.5 + exp * 15.0;
            gene_dispersion[gene_idx] = 0.1 + u * 0.6;
        }
    }

    for cell_idx in 0..nrow {
        let mut rng = StdRng::seed_from_u64(seed as u64 + cell_idx as u64);
        let cell_type = cell_idx % n_cell_types;
        let batch = (cell_idx * n_batches) / nrow;

        cell_type_labels.push(cell_type);
        batch_labels.push(batch);

        temp_vec.clear();

        for gene_idx in 0..ncol {
            let mut mu = gene_base_mean[gene_idx];

            if let Some(&marker_ct) = marker_to_celltype.get(&gene_idx) {
                if marker_ct == cell_type {
                    mu *= 4.0;
                } else {
                    mu *= 0.3;
                }
            }

            // Apply batch effect
            mu *= batch_effect[batch][gene_idx];

            // Global coherent batch shift (creates separation in expression space)
            if batch > 0 {
                mu *= 1.0 + (batch as f64) * systematic_mult;
            }

            // Cap to prevent explosion
            mu = mu.min(50.0);

            // Batch-specific gene module effects
            if batch > 0 {
                let module = gene_idx / 100;
                if module % n_batches == batch {
                    mu *= module_mult;
                }
            }

            // Final cap to keep Poisson sampler fast
            mu = mu.min(100.0);

            let p = gene_dispersion[gene_idx] / (gene_dispersion[gene_idx] + mu);
            let r = gene_dispersion[gene_idx];

            let shape = r;
            let scale = (1.0 - p) / p;
            let gamma_sample = gamma_sample(&mut rng, shape, scale);
            let lambda = gamma_sample;
            let count = poisson_sample(&mut rng, lambda);

            if count > 0 {
                temp_vec.push((gene_idx, count));
            }
        }

        temp_vec.sort_unstable_by_key(|(gene_idx, _)| *gene_idx);

        for &(gene_idx, count) in &temp_vec {
            indices.push(gene_idx);
            data.push(count);
        }

        indptr.push(indices.len());
    }

    let csr = CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None::<Vec<u32>>,
        shape: (nrow, ncol),
    };

    (csr, cell_type_labels, batch_labels)
}

/// Generate sample labels with configurable cell type bias
///
/// ### Params
///
/// * `cell_type_labels` - Vector of cell type assignments
/// * `n_samples` - Number of samples to generate
/// * `bias` - Level of bias in cell type distribution across samples
/// * `seed` - Integer for reproducibility
///
/// ### Returns
///
/// Vector of sample labels with biased cell type distributions
pub fn generate_sample_labels(
    cell_type_labels: &[usize],
    n_samples: usize,
    bias: &SampleBias,
    seed: usize,
) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let n_cells = cell_type_labels.len();
    let n_cell_types = cell_type_labels.iter().max().map(|&x| x + 1).unwrap_or(0);

    let mut sample_labels: Vec<usize> = Vec::with_capacity(n_cells);

    for &cell_type in cell_type_labels {
        let sample = match bias {
            SampleBias::Even => {
                // uniform random assignment
                (rng.random::<f64>() * n_samples as f64).floor() as usize
            }
            SampleBias::SlightlyUneven => {
                // mild preference for certain samples based on cell type
                let mut weights = vec![1.0; n_samples];
                for s in 0..n_samples {
                    let s_norm = s as f64 / (n_samples.max(1) - 1) as f64;
                    let ct_norm = cell_type as f64 / (n_cell_types.max(1) - 1) as f64;
                    let diff = (s_norm - ct_norm).abs();
                    weights[s] = (-diff).exp();
                }

                let sum: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= sum;
                }

                let u: f64 = rng.random();
                let mut cumulative = 0.0;
                let mut sample = 0;
                for (s, &weight) in weights.iter().enumerate() {
                    cumulative += weight;
                    if u <= cumulative {
                        sample = s;
                        break;
                    }
                }
                sample
            }
            SampleBias::VeryUneven => {
                // strong preference for certain samples based on cell type
                let mut weights = vec![0.1; n_samples];
                for s in 0..n_samples {
                    let s_norm = s as f64 / (n_samples.max(1) - 1) as f64;
                    let ct_norm = cell_type as f64 / (n_cell_types.max(1) - 1) as f64;
                    let diff = (s_norm - ct_norm).abs();
                    weights[s] = (-4.0 * diff).exp();
                }

                let sum: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= sum;
                }

                let u: f64 = rng.random();
                let mut cumulative = 0.0;
                let mut sample = 0;
                for (s, &weight) in weights.iter().enumerate() {
                    cumulative += weight;
                    if u <= cumulative {
                        sample = s;
                        break;
                    }
                }
                sample
            }
        };

        sample_labels.push(sample.min(n_samples - 1));
    }

    sample_labels
}

/// Helper function to create synthetic ADT (protein) counts
///
/// Produces a dense cells x proteins matrix of raw ADT counts with the
/// structure CLR and DSB normalisation care about: every protein carries an
/// ambient background, marker proteins are additionally elevated in their
/// target cell type, and isotype controls only ever carry background. A
/// per-cell size factor models capture efficiency and an optional per-batch
/// per-protein multiplier models staining differences across batches.
///
/// Cell type and batch assignment use the same formulas as
/// [create_celltype_sparse_csr_data], so labels line up cell-for-cell when the
/// two are generated with matching `nrow`, number of cell types and
/// `n_batches`.
///
/// ### Params
///
/// * `nrow` - Number of rows (cells).
/// * `n_proteins` - Number of columns (proteins) in the panel.
/// * `cell_type_configs` - Marker protein column indices per cell type.
/// * `isotype_controls` - Column indices of the isotype controls.
/// * `n_batches` - Number of batches to introduce.
/// * `batch_effect_strength` - Strength of the per-batch staining effect.
/// * `seed` - Integer for reproducibility.
///
/// ### Returns
///
/// A tuple `(counts, cell_type_labels, batch_labels)` where `counts` is a
/// row-major dense matrix of length `nrow * n_proteins`.
pub fn create_adt_synthetic_data(
    nrow: usize,
    n_proteins: usize,
    cell_type_configs: Vec<CellTypeConfig>,
    isotype_controls: Vec<usize>,
    n_batches: usize,
    batch_effect_strength: &str,
    seed: usize,
) -> (Vec<u32>, Vec<usize>, Vec<usize>) {
    let batch_strength =
        parse_batch_effect_strength(batch_effect_strength).unwrap_or(BatchEffectStrength::Strong);

    let n_cell_types = cell_type_configs.len();

    let batch_spread = match batch_strength {
        BatchEffectStrength::Weak => 0.1,
        BatchEffectStrength::Medium => 0.3,
        BatchEffectStrength::Strong => 0.6,
    };

    let mut param_rng = StdRng::seed_from_u64(seed as u64);

    // per-protein parameters
    let mut bg_mean = vec![0.0; n_proteins];
    let mut signal_mean = vec![0.0; n_proteins];
    let mut dispersion = vec![0.0; n_proteins];
    for p in 0..n_proteins {
        bg_mean[p] = 1.0 + param_rng.random::<f64>() * 9.0; // ambient 1-10
        signal_mean[p] = 300.0 + param_rng.random::<f64>() * 1700.0; // specific 300-2000 (markers only)
        dispersion[p] = 3.0 + param_rng.random::<f64>() * 12.0; // NB size r
    }

    // per-batch per-protein staining multiplier (batch 0 == 1.0)
    let mut batch_mult = vec![vec![1.0; n_proteins]; n_batches.max(1)];
    for b in 1..n_batches {
        for p in 0..n_proteins {
            let u: f64 = param_rng.random();
            batch_mult[b][p] = (1.0 + (u * 2.0 - 1.0) * batch_spread).max(0.05);
        }
    }

    // marker protein -> owning cell type
    let mut marker_to_celltype = FxHashMap::default();
    for (ct_idx, config) in cell_type_configs.iter().enumerate() {
        for &protein_idx in &config.marker_genes {
            marker_to_celltype.insert(protein_idx, ct_idx);
        }
    }
    // isotypes carry background only; never let them act as markers
    for &iso in &isotype_controls {
        marker_to_celltype.remove(&iso);
    }

    let mut counts = vec![0u32; nrow * n_proteins];
    let mut cell_type_labels = Vec::with_capacity(nrow);
    let mut batch_labels = Vec::with_capacity(nrow);

    for cell_idx in 0..nrow {
        let mut rng = StdRng::seed_from_u64(seed as u64 + cell_idx as u64);
        let cell_type = cell_idx % n_cell_types;
        let batch = (cell_idx * n_batches) / nrow;

        cell_type_labels.push(cell_type);
        batch_labels.push(batch);

        // per-cell capture efficiency
        let z: f64 = rng.sample(StandardNormal);
        let size_factor = (0.3 * z).exp();

        let row = cell_idx * n_proteins;
        for p in 0..n_proteins {
            let mut mu = bg_mean[p];

            if let Some(&marker_ct) = marker_to_celltype.get(&p)
                && marker_ct == cell_type
            {
                mu += signal_mean[p];
            }

            mu *= size_factor;
            mu *= batch_mult[batch][p];

            let r = dispersion[p];
            let scale = mu / r;
            let lambda = gamma_sample(&mut rng, r, scale);
            counts[row + p] = poisson_sample(&mut rng, lambda);
        }
    }

    (counts, cell_type_labels, batch_labels)
}

/////////////////////////////
// DIALOGUE synthetic data //
/////////////////////////////

////////////
// Consts //
////////////

/// Gaussian noise added on top of a sample-level feature component.
const DIALOGUE_FEATURE_NOISE: f64 = 0.45;

/// Baseline expression level of a planted gene, before the programme moves it.
const DIALOGUE_PLANTED_BASE: f64 = 1.2;

/// How hard the programme drives a planted gene.
const DIALOGUE_PLANTED_EFFECT: f64 = 0.9;

/// Floor on a planted gene's mean, so a strongly negative programme score
/// cannot push it to zero.
const DIALOGUE_MEAN_FLOOR: f64 = 0.05;

/// Expression level of everything that is not planted.
const DIALOGUE_BACKGROUND_MEAN: f64 = 0.6;

/// Uniform draw below which an entry is dropped, giving the matrix its
/// sparsity. Roughly 55% of entries survive.
const DIALOGUE_DROPOUT: f64 = 0.45;

/// Offset on the surviving draw, so a kept entry never scales its mean by zero.
const DIALOGUE_DRAW_OFFSET: f64 = 0.5;

/// Scales the normalised value into the raw count layer.
const DIALOGUE_COUNT_SCALE: f32 = 10.0;

/// Shape of a synthetic DIALOGUE experiment.
///
/// Cells are laid out contiguously by cell type, and within a cell type
/// contiguously by sample, so every cell type ends up with the same sample
/// composition. That is the easy case for the method: it is a fixture for
/// testing the pipeline, not a stress test of the sample overlap logic.
#[derive(Clone, Copy, Debug)]
pub struct DialogueSyntheticParams {
    /// Samples the experiment spans. Below `MIN_SHARED_SAMPLES` the
    /// decomposition refuses to run.
    pub n_samples: usize,
    /// Cells per sample per cell type.
    pub cells_per_sample: usize,
    /// Cell types. DIALOGUE needs at least two.
    pub n_cell_types: usize,
    /// Feature columns per cell type.
    pub n_features: usize,
    /// Feature columns carrying a per-sample component. Column zero of those is
    /// the shared programme, the rest are cell-type-specific nuisance. Anything
    /// past this count is pure noise, and exists so the ANOVA filter has
    /// something to reject.
    pub n_sample_features: usize,
    /// Genes in the store.
    pub n_genes: usize,
    /// Planted genes per cell type. Cell type `t` owns genes
    /// `t * n_planted .. (t + 1) * n_planted`, so the blocks have to fit in
    /// `n_genes`.
    pub n_planted: usize,
}

impl DialogueSyntheticParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Samples the experiment spans
    /// * `cells_per_sample` - Cells per sample per cell type
    /// * `n_cell_types` - Cell types
    /// * `n_features` - Feature columns per cell type
    /// * `n_sample_features` - Feature columns carrying a per-sample component
    /// * `n_genes` - Genes in the store
    /// * `n_planted` - Planted genes per cell type
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here, see
    /// [DialogueSyntheticParams::validate].
    pub fn new(
        n_samples: usize,
        cells_per_sample: usize,
        n_cell_types: usize,
        n_features: usize,
        n_sample_features: usize,
        n_genes: usize,
        n_planted: usize,
    ) -> Self {
        Self {
            n_samples,
            cells_per_sample,
            n_cell_types,
            n_features,
            n_sample_features,
            n_genes,
            n_planted,
        }
    }

    /// Checks the shape is buildable.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [BixverseErrors::InvalidArgument] describing the first
    /// problem found.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        if self.n_cell_types < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "DIALOGUE needs at least two cell types; got {}.",
                self.n_cell_types
            )));
        }
        if self.n_samples == 0 || self.cells_per_sample == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "n_samples and cells_per_sample must both be positive.".to_string(),
            ));
        }
        if self.n_features < 2 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "DIALOGUE needs at least two feature columns; got {}.",
                self.n_features
            )));
        }
        if self.n_sample_features == 0 || self.n_sample_features > self.n_features {
            return Err(BixverseErrors::InvalidArgument(format!(
                "n_sample_features must lie in 1..={}; got {}.",
                self.n_features, self.n_sample_features
            )));
        }
        if self.n_planted * self.n_cell_types > self.n_genes {
            return Err(BixverseErrors::InvalidArgument(format!(
                "the planted blocks need {} genes but only {} exist.",
                self.n_planted * self.n_cell_types,
                self.n_genes
            )));
        }
        Ok(())
    }
}

impl Default for DialogueSyntheticParams {
    fn default() -> Self {
        Self {
            n_samples: 14,
            cells_per_sample: 25,
            n_cell_types: 3,
            n_features: 8,
            n_sample_features: 5,
            n_genes: 90,
            n_planted: 8,
        }
    }
}

/// A synthetic DIALOGUE experiment, plus the ground truth to check against.
#[derive(Clone, Debug)]
pub struct DialogueSyntheticData {
    /// Counts, CSC with shape (cells, genes). Raw counts in `data`, the
    /// normalised layer they were scaled from in `data_2`.
    pub matrix: CompressedSparseData2<u32, f32>,
    /// Global cell indices per cell type.
    pub cell_type_indices: Vec<Vec<usize>>,
    /// Cell-level features per cell type, rows aligned to
    /// `cell_type_indices`.
    pub features: Vec<Mat<f64>>,
    /// Sample code per global cell.
    pub sample_ids: Vec<usize>,
    /// Quality covariate per global cell. Pure noise: nothing in the data
    /// depends on it, so anything a method attributes to it is spurious.
    pub quality: Vec<f64>,
    /// Per-sample latent the planted programme follows.
    pub latent: Vec<f64>,
    /// Planted gene indices per cell type.
    pub planted: Vec<Vec<usize>>,
}

/// Builds a synthetic experiment with one planted multicellular programme.
///
/// Every cell type gets its own noise and its own sample-level nuisance
/// factors; only feature zero and the planted genes carry the shared latent, so
/// anything a method finds beyond that is something it invented.
///
/// The count layer is a scaled copy of the normalised layer rather than a draw
/// from a count model. That is deliberate: the point of the fixture is a clean
/// planted signal, and a gamma-Poisson draw on top would blur it.
///
/// ### Params
///
/// * `params` - See [DialogueSyntheticParams]
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// The [DialogueSyntheticData], or the first shape problem found.
pub fn create_dialogue_synthetic_data(
    params: &DialogueSyntheticParams,
    seed: u64,
) -> Result<DialogueSyntheticData, BixverseErrors> {
    params.validate()?;

    let DialogueSyntheticParams {
        n_samples,
        cells_per_sample,
        n_cell_types,
        n_features,
        n_sample_features,
        n_genes,
        n_planted,
    } = *params;

    let mut rng = StdRng::seed_from_u64(seed);

    let latent: Vec<f64> = (0..n_samples).map(|_| standard_normal(&mut rng)).collect();
    // Sample-level nuisance, one set per cell type per feature slot.
    let nuisance: Vec<Vec<Vec<f64>>> = (0..n_cell_types)
        .map(|_| {
            (0..n_sample_features)
                .map(|_| (0..n_samples).map(|_| standard_normal(&mut rng)).collect())
                .collect()
        })
        .collect();

    let per_type = n_samples * cells_per_sample;
    let n_cells = per_type * n_cell_types;

    let mut sample_ids = vec![0usize; n_cells];
    let mut quality = vec![0.0_f64; n_cells];
    let mut cell_type_indices: Vec<Vec<usize>> = Vec::with_capacity(n_cell_types);
    let mut features: Vec<Mat<f64>> = Vec::with_capacity(n_cell_types);
    // Per-cell programme strength, used to drive the planted genes.
    let mut strength = vec![0.0_f64; n_cells];

    let mut cursor = 0usize;
    for t in 0..n_cell_types {
        let mut cells = Vec::with_capacity(per_type);
        let mut feature = Mat::<f64>::zeros(per_type, n_features);
        for s in 0..n_samples {
            for _ in 0..cells_per_sample {
                let global = cursor;
                let local = cells.len();
                cells.push(global);
                sample_ids[global] = s;
                quality[global] = standard_normal(&mut rng);

                // Feature 0 is the shared programme; 1..n_sample_features are
                // cell-type-specific sample effects; the rest are noise.
                let shared = latent[s] + DIALOGUE_FEATURE_NOISE * standard_normal(&mut rng);
                feature[(local, 0)] = shared;
                strength[global] = shared;
                for f in 1..n_sample_features {
                    feature[(local, f)] =
                        nuisance[t][f][s] + DIALOGUE_FEATURE_NOISE * standard_normal(&mut rng);
                }
                for f in n_sample_features..n_features {
                    feature[(local, f)] = standard_normal(&mut rng);
                }
                cursor += 1;
            }
        }
        cell_type_indices.push(cells);
        features.push(feature);
    }

    // Genes: a planted block per cell type, then noise. The planted genes are
    // driven by that cell type's own programme strength.
    let planted: Vec<Vec<usize>> = (0..n_cell_types)
        .map(|t| (t * n_planted..(t + 1) * n_planted).collect())
        .collect();

    let unit = Uniform::new(0.0_f64, 1.0).expect("valid range");
    let mut data: Vec<u32> = Vec::new();
    let mut data_norm: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut indptr: Vec<u32> = vec![0];
    // CSC over genes, shape (cells, genes). Cells are laid out contiguously by
    // cell type above, so `cell / per_type` gives the owner directly.
    for gene in 0..n_genes {
        let owner = planted.iter().position(|p| p.contains(&gene));
        for cell in 0..n_cells {
            let mean = if owner == Some(cell / per_type) {
                // Planted: expression rises with the programme.
                (DIALOGUE_PLANTED_BASE + DIALOGUE_PLANTED_EFFECT * strength[cell])
                    .max(DIALOGUE_MEAN_FLOOR)
            } else {
                DIALOGUE_BACKGROUND_MEAN
            };
            // Sparse and non-negative, standing in for a normalised count.
            let draw: f64 = rng.sample(unit);
            if draw >= DIALOGUE_DROPOUT {
                let value = (mean * (DIALOGUE_DRAW_OFFSET + draw)).max(0.0) as f32;
                if value > 0.0 {
                    data.push((value * DIALOGUE_COUNT_SCALE).round() as u32);
                    data_norm.push(value);
                    indices.push(cell as u32);
                }
            }
        }
        indptr.push(indices.len() as u32);
    }

    let matrix = CompressedSparseData2::new_csc(
        &data,
        &indices,
        &indptr,
        Some(&data_norm),
        (n_cells, n_genes),
    );

    Ok(DialogueSyntheticData {
        matrix,
        cell_type_indices,
        features,
        sample_ids,
        quality,
        latent,
        planted,
    })
}

//////////////////////////////
// CellSweep synthetic data //
//////////////////////////////

////////////
// Consts //
////////////

/// Gamma shape the per-gene base expression is drawn from. Below one the draw
/// piles up near zero and the fixture ends up with dead genes.
const CELLSWEEP_BASE_SHAPE: f64 = 1.5;

/// Lower clamp on a planted ambient fraction, so no barcode is clean.
const CELLSWEEP_ALPHA_MIN: f64 = 0.02;

/// Upper clamp on a planted ambient fraction. Sits below the `alpha_cap` the
/// model defaults to, so the truth is inside what the EM can reach.
const CELLSWEEP_ALPHA_MAX: f64 = 0.85;

/// Shape of a synthetic CellSweep experiment.
///
/// Real barcodes come first, empty droplets after, which is the order
/// `run_cellsweep` wants its per-sample index lists in anyway. Cell types are
/// assigned round-robin over the real barcodes.
#[derive(Clone, Copy, Debug)]
pub struct CellSweepSyntheticParams {
    /// Number of real barcodes.
    pub n_real: usize,
    /// Number of empty droplets. The ambient profile is estimated off these,
    /// so too few makes the fixture noisy rather than wrong.
    pub n_empty: usize,
    /// Number of genes.
    pub n_genes: usize,
    /// Number of cell types.
    pub n_celltypes: usize,
    /// Width of each cell type's marker block. Blocks are disjoint and laid
    /// out from gene zero.
    pub n_markers: usize,
    /// How much a marker gene is enriched over background in its own cell
    /// type's profile.
    pub marker_weight: f64,
    /// Fraction of the soup coming from cell type zero, the population that
    /// lyses. The remainder is a background of its own, which is what keeps
    /// the ambient profile out of the span of the cell type profiles and the
    /// contamination fraction identifiable.
    pub ambient_dominance: f64,
    /// Mean planted ambient fraction across real barcodes.
    pub alpha_mean: f64,
    /// Spread of the planted ambient fraction.
    pub alpha_sd: f64,
    /// Expected library size of a real barcode.
    pub real_lib_size: usize,
    /// Expected library size of an empty droplet.
    pub empty_lib_size: usize,
}

impl Default for CellSweepSyntheticParams {
    fn default() -> Self {
        Self {
            n_real: 600,
            n_empty: 2000,
            n_genes: 200,
            n_celltypes: 3,
            n_markers: 20,
            marker_weight: 25.0,
            ambient_dominance: 0.6,
            alpha_mean: 0.3,
            alpha_sd: 0.12,
            real_lib_size: 3000,
            empty_lib_size: 120,
        }
    }
}

impl CellSweepSyntheticParams {
    /// Checks the shape is buildable.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [BixverseErrors::InvalidArgument] describing the first
    /// problem found.
    pub fn validate(&self) -> Result<(), BixverseErrors> {
        if self.n_real == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "n_real must be positive.".to_string(),
            ));
        }
        if self.n_empty < MIN_EMPTY_DROPLETS {
            return Err(BixverseErrors::InvalidArgument(format!(
                "CellSweep needs at least {} empty droplets; got {}.",
                MIN_EMPTY_DROPLETS, self.n_empty
            )));
        }
        if self.n_celltypes == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "n_celltypes must be positive.".to_string(),
            ));
        }
        if self.n_markers == 0 || self.n_markers * self.n_celltypes > self.n_genes {
            return Err(BixverseErrors::InvalidArgument(format!(
                "the marker blocks need {} genes but only {} exist.",
                self.n_markers * self.n_celltypes,
                self.n_genes
            )));
        }
        if self.marker_weight <= 1.0 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "marker_weight must exceed 1; got {}.",
                self.marker_weight
            )));
        }
        if !(0.0..=1.0).contains(&self.ambient_dominance) {
            return Err(BixverseErrors::InvalidArgument(format!(
                "ambient_dominance must lie in [0, 1]; got {}.",
                self.ambient_dominance
            )));
        }
        if !(CELLSWEEP_ALPHA_MIN..=CELLSWEEP_ALPHA_MAX).contains(&self.alpha_mean) {
            return Err(BixverseErrors::InvalidArgument(format!(
                "alpha_mean must lie in [{}, {}]; got {}.",
                CELLSWEEP_ALPHA_MIN, CELLSWEEP_ALPHA_MAX, self.alpha_mean
            )));
        }
        if self.alpha_sd < 0.0 {
            return Err(BixverseErrors::InvalidArgument(format!(
                "alpha_sd must not be negative; got {}.",
                self.alpha_sd
            )));
        }
        if self.real_lib_size == 0 || self.empty_lib_size == 0 {
            return Err(BixverseErrors::InvalidArgument(
                "real_lib_size and empty_lib_size must both be positive.".to_string(),
            ));
        }
        Ok(())
    }
}

/// A synthetic CellSweep experiment, plus the ground truth to check against.
#[derive(Clone, Debug)]
pub struct CellSweepSyntheticData {
    /// Counts, CSR with shape (cells, genes). Real barcodes first, then the
    /// empty droplets.
    pub matrix: CompressedSparseData2<u32>,
    /// Cell type per real barcode. Empty droplets have none.
    pub cell_type_indices: Vec<usize>,
    /// Empty droplet mask over all barcodes.
    pub is_empty: Vec<bool>,
    /// Planted ambient fraction per real barcode.
    pub alpha_true: Vec<f64>,
    /// The soup the empty droplets were drawn from. Sums to one.
    pub ambient_true: Vec<f64>,
    /// Cell type profiles, row-major `n_celltypes x n_genes`. Each row sums to
    /// one.
    pub celltype_profiles_true: Vec<f64>,
}

/// Builds a synthetic experiment with a planted ambient profile.
///
/// Every real barcode is a two-component multinomial: a planted fraction
/// `alpha_i` of its library is drawn from the soup, the rest from its own cell
/// type profile. Empty droplets are pure soup at a much smaller library size,
/// which is the signal the ambient profile is estimated off.
///
/// The soup is cell type zero plus a background of its own rather than a
/// mixture of every profile. A soup that sits in the span of the cell type
/// profiles makes the contamination fraction unidentifiable, and the fixture
/// would then be testing the repulsion term rather than the model.
///
/// ### Params
///
/// * `params` - See [CellSweepSyntheticParams]
/// * `seed` - Seed for reproducibility
///
/// ### Returns
///
/// The [CellSweepSyntheticData], or the first shape problem found.
pub fn create_cellsweep_synthetic_data(
    params: &CellSweepSyntheticParams,
    seed: u64,
) -> Result<CellSweepSyntheticData, BixverseErrors> {
    params.validate()?;

    let CellSweepSyntheticParams {
        n_real,
        n_empty,
        n_genes,
        n_celltypes,
        n_markers,
        marker_weight,
        ambient_dominance,
        alpha_mean,
        alpha_sd,
        real_lib_size,
        empty_lib_size,
    } = *params;

    let mut rng = StdRng::seed_from_u64(seed);

    // Every gene gets a base expression level, shared across cell types, so
    // the profiles differ from each other only in their marker blocks. A flat
    // base would leave the soup a two-valued step function and nothing
    // downstream could be ranked against it.
    let base: Vec<f64> = (0..n_genes)
        .map(|_| gamma_sample(&mut rng, CELLSWEEP_BASE_SHAPE, 1.0))
        .collect();

    let mut profiles = vec![0.0_f64; n_celltypes * n_genes];
    for t in 0..n_celltypes {
        let row = &mut profiles[t * n_genes..(t + 1) * n_genes];
        for (gene, weight) in row.iter_mut().enumerate() {
            *weight = if gene >= t * n_markers && gene < (t + 1) * n_markers {
                marker_weight * base[gene]
            } else {
                base[gene]
            };
        }
        let total: f64 = row.iter().sum();
        row.iter_mut().for_each(|w| *w /= total);
    }

    // Soup: the lysing population plus a background drawn independently of the
    // cell type profiles.
    let mut background: Vec<f64> = (0..n_genes)
        .map(|_| gamma_sample(&mut rng, CELLSWEEP_BASE_SHAPE, 1.0))
        .collect();
    let total: f64 = background.iter().sum();
    background.iter_mut().for_each(|w| *w /= total);

    let ambient: Vec<f64> = (0..n_genes)
        .map(|gene| {
            ambient_dominance * profiles[gene] + (1.0 - ambient_dominance) * background[gene]
        })
        .collect();

    let ambient_cdf = to_cdf(&ambient);
    let profile_cdfs: Vec<Vec<f64>> = (0..n_celltypes)
        .map(|t| to_cdf(&profiles[t * n_genes..(t + 1) * n_genes]))
        .collect();

    let n_cells = n_real + n_empty;
    let mut indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    let mut indices: Vec<usize> = Vec::with_capacity(n_cells * 64);
    let mut data: Vec<u32> = Vec::with_capacity(n_cells * 64);
    indptr.push(0);

    let mut cell_type_indices = Vec::with_capacity(n_real);
    let mut alpha_true = Vec::with_capacity(n_real);
    let mut is_empty = vec![false; n_cells];
    let mut counts = vec![0_u32; n_genes];

    for cell in 0..n_cells {
        counts.iter_mut().for_each(|c| *c = 0);

        if cell < n_real {
            let cell_type = cell % n_celltypes;
            let alpha = (alpha_mean + alpha_sd * standard_normal(&mut rng))
                .clamp(CELLSWEEP_ALPHA_MIN, CELLSWEEP_ALPHA_MAX);
            let lib = poisson_sample(&mut rng, real_lib_size as f64);
            for _ in 0..lib {
                let cdf = if rng.random::<f64>() < alpha {
                    &ambient_cdf
                } else {
                    &profile_cdfs[cell_type]
                };
                counts[draw_from_cdf(&mut rng, cdf)] += 1;
            }
            cell_type_indices.push(cell_type);
            alpha_true.push(alpha);
        } else {
            is_empty[cell] = true;
            let lib = poisson_sample(&mut rng, empty_lib_size as f64);
            for _ in 0..lib {
                counts[draw_from_cdf(&mut rng, &ambient_cdf)] += 1;
            }
        }

        for (gene, &count) in counts.iter().enumerate() {
            if count > 0 {
                indices.push(gene);
                data.push(count);
            }
        }
        indptr.push(indices.len());
    }

    let matrix = CompressedSparseData2 {
        data,
        indices: indices.index_cast(),
        indptr: indptr.index_cast(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None::<Vec<u32>>,
        shape: (n_cells, n_genes),
    };

    Ok(CellSweepSyntheticData {
        matrix,
        cell_type_indices,
        is_empty,
        alpha_true,
        ambient_true: ambient,
        celltype_profiles_true: profiles,
    })
}

/////////////
// Helpers //
/////////////

/// Helper function to draw from the standard normal
///
/// A free function rather than a closure, so the borrow on the generator ends
/// at the call and the caller can keep drawing from other distributions.
///
/// ### Params
///
/// * `rng` - The random number generator
///
/// ### Returns
///
/// A sample from `N(0, 1)`
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    rng.sample(StandardNormal)
}

/// Helper function to sample from a Gamma distribution
///
/// Uses the Marsaglia and Tsang method for shape >= 1, with Ahrens-Dieter
/// method for shape < 1.
///
/// ### Params
///
/// * `rng` - Random number generator
/// * `shape` - Shape parameter (k or α)
/// * `scale` - Scale parameter (θ)
///
/// ### Returns
///
/// A sample from Gamma(shape, scale)
fn gamma_sample<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.random::<f64>();
        return gamma_sample(rng, 1.0 + shape, scale) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = rng.sample(StandardNormal);
        let v = (1.0 + c * x).powi(3);

        if v > 0.0 {
            let u = rng.random::<f64>();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod cellsweep_synthetic_tests {
    use super::*;

    /// Pooled counts per gene over the rows in `rows`.
    fn pooled(data: &CellSweepSyntheticData, rows: &[usize]) -> Vec<f64> {
        let n_genes = data.matrix.shape.1;
        let mut out = vec![0.0; n_genes];
        for &row in rows {
            let start = data.matrix.indptr[row] as usize;
            let end = data.matrix.indptr[row + 1] as usize;
            for k in start..end {
                out[data.matrix.indices[k] as usize] += data.matrix.data[k] as f64;
            }
        }
        out
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
        cov / (va * vb)
    }

    #[test]
    fn test_synthetic_shape_and_empty_layout() {
        let params = CellSweepSyntheticParams::default();
        let data = create_cellsweep_synthetic_data(&params, 42).unwrap();

        assert_eq!(
            data.matrix.shape,
            (params.n_real + params.n_empty, params.n_genes)
        );
        assert_eq!(data.matrix.indptr.len(), params.n_real + params.n_empty + 1);
        assert_eq!(data.cell_type_indices.len(), params.n_real);
        assert_eq!(data.alpha_true.len(), params.n_real);
        assert_eq!(data.ambient_true.len(), params.n_genes);
        assert_eq!(
            data.celltype_profiles_true.len(),
            params.n_celltypes * params.n_genes
        );
        assert!(data.is_empty[..params.n_real].iter().all(|e| !e));
        assert!(data.is_empty[params.n_real..].iter().all(|e| *e));
    }

    #[test]
    fn test_synthetic_is_deterministic() {
        let params = CellSweepSyntheticParams::default();
        let a = create_cellsweep_synthetic_data(&params, 7).unwrap();
        let b = create_cellsweep_synthetic_data(&params, 7).unwrap();
        let c = create_cellsweep_synthetic_data(&params, 8).unwrap();

        assert_eq!(a.matrix.data, b.matrix.data);
        assert_eq!(a.matrix.indices, b.matrix.indices);
        assert_eq!(a.alpha_true, b.alpha_true);
        assert_ne!(a.matrix.data, c.matrix.data);
    }

    #[test]
    fn test_profiles_and_ambient_are_distributions() {
        let params = CellSweepSyntheticParams::default();
        let data = create_cellsweep_synthetic_data(&params, 42).unwrap();

        assert!((data.ambient_true.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for t in 0..params.n_celltypes {
            let row = &data.celltype_profiles_true[t * params.n_genes..(t + 1) * params.n_genes];
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert!(data.alpha_true.iter().all(|a| *a > 0.0 && *a < 1.0));
    }

    #[test]
    fn test_empty_droplets_reproduce_the_soup() {
        let params = CellSweepSyntheticParams::default();
        let data = create_cellsweep_synthetic_data(&params, 42).unwrap();

        let empties: Vec<usize> = (params.n_real..params.n_real + params.n_empty).collect();
        let observed = pooled(&data, &empties);

        assert!(pearson(&observed, &data.ambient_true) > 0.99);
    }

    #[test]
    fn test_contamination_scales_with_alpha() {
        let base = CellSweepSyntheticParams::default();
        let heavy = CellSweepSyntheticParams {
            alpha_mean: 0.7,
            ..base
        };

        // Cell type one barcodes: they own markers n_markers..2*n_markers, so
        // anything they carry in cell type zero's block came from the soup.
        let soup_block = 0..base.n_markers;
        let leak = |params: &CellSweepSyntheticParams| {
            let data = create_cellsweep_synthetic_data(params, 42).unwrap();
            let rows: Vec<usize> = (0..params.n_real)
                .filter(|c| c % params.n_celltypes == 1)
                .collect();
            let counts = pooled(&data, &rows);
            let total: f64 = counts.iter().sum();
            counts[soup_block.clone()].iter().sum::<f64>() / total
        };

        assert!(leak(&heavy) > 2.0 * leak(&base));
    }

    #[test]
    fn test_validate_rejects_bad_shapes() {
        let too_few_empties = CellSweepSyntheticParams {
            n_empty: MIN_EMPTY_DROPLETS - 1,
            ..Default::default()
        };
        assert!(create_cellsweep_synthetic_data(&too_few_empties, 42).is_err());

        let markers_do_not_fit = CellSweepSyntheticParams {
            n_markers: 100,
            n_celltypes: 3,
            n_genes: 200,
            ..Default::default()
        };
        assert!(create_cellsweep_synthetic_data(&markers_do_not_fit, 42).is_err());

        let no_marker_signal = CellSweepSyntheticParams {
            marker_weight: 1.0,
            ..Default::default()
        };
        assert!(create_cellsweep_synthetic_data(&no_marker_signal, 42).is_err());
    }
}
