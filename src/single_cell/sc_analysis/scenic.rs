use faer::Mat;
use indexmap::IndexSet;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::time::Instant;

use crate::prelude::*;
use crate::utils::simd::sum_simd_f32;

const SCENIC_GENE_CHUNK_SIZE: usize = 1000;

///////////
// Enums //
///////////

/// Enum to define the type of Regression learner to use
#[derive(Clone, Debug)]
pub enum RegressionLearner {
    ExtraTrees(ExtraTreesConfig),
    RandomForest(RandomForestConfig),
}

/// Default implementation for RegressionLearner
impl Default for RegressionLearner {
    fn default() -> Self {
        RegressionLearner::ExtraTrees(ExtraTreesConfig::default())
    }
}

/// Parse the regression learner
pub fn parse_regression_learner(s: &str) -> Option<RegressionLearner> {
    match s.to_lowercase().as_str() {
        "extratrees" => Some(RegressionLearner::ExtraTrees(ExtraTreesConfig::default())),
        "rf" | "randomforest" => Some(RegressionLearner::RandomForest(
            RandomForestConfig::default(),
        )),
        _ => None,
    }
}

//////////////////
// Shared trait //
//////////////////

trait TreeRegressorConfig: Sync {
    fn n_trees(&self) -> usize;
    fn min_samples_leaf(&self) -> usize;
    fn n_features_split(&self) -> usize;
    fn random_threshold(&self) -> bool;
    fn subsample_rate(&self) -> f32 {
        1.0
    }
    fn bootstrap(&self) -> bool {
        false
    }
    fn max_depth(&self) -> Option<usize> {
        None
    }
    fn min_variance(&self) -> f32 {
        1e-10
    }
    fn n_thresholds(&self) -> usize {
        1
    }
}

////////////
// Params //
////////////

////////////////
// ExtraTrees //
////////////////

#[derive(Clone, Debug)]
pub struct ExtraTreesConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub n_thresholds: usize,
    pub max_depth: Option<usize>,
}

impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            min_samples_leaf: 50,
            n_features_split: 0,
            n_thresholds: 1,
            max_depth: Some(10), // Reduced depth
        }
    }
}

impl TreeRegressorConfig for ExtraTreesConfig {
    fn n_trees(&self) -> usize {
        self.n_trees
    }
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }
    fn n_features_split(&self) -> usize {
        self.n_features_split
    }
    fn random_threshold(&self) -> bool {
        true
    }
    fn n_thresholds(&self) -> usize {
        self.n_thresholds
    }
    fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }
}

//////////////////
// RandomForest //
//////////////////

#[derive(Clone, Debug)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub n_features_split: usize,
    pub subsample_rate: f32,
    pub bootstrap: bool,
    pub max_depth: Option<usize>,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            min_samples_leaf: 50,
            n_features_split: 0,
            subsample_rate: 0.632,
            bootstrap: false,
            max_depth: Some(10), // Reduced depth
        }
    }
}

impl TreeRegressorConfig for RandomForestConfig {
    fn n_trees(&self) -> usize {
        self.n_trees
    }
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }
    fn n_features_split(&self) -> usize {
        self.n_features_split
    }
    fn random_threshold(&self) -> bool {
        false
    }
    fn subsample_rate(&self) -> f32 {
        self.subsample_rate
    }
    fn bootstrap(&self) -> bool {
        self.bootstrap
    }
    fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }
}

/////////////////////
// Storage helpers //
/////////////////////

///////////////
// Quantiser //
///////////////

pub struct DenseQuantisedStore {
    data: Vec<u8>,
    n_cells: usize,
    pub n_features: usize,
    feature_min: Vec<f32>,
    feature_range: Vec<f32>,
}

impl DenseQuantisedStore {
    pub fn from_csc(mat: &CompressedSparseData<u16, f32>, n_cells: usize) -> Self {
        let n_features = mat.indptr.len() - 1;
        let mut data = vec![0u8; n_features * n_cells];
        let mut mins = Vec::with_capacity(n_features);
        let mut ranges = Vec::with_capacity(n_features);

        let vals = mat.data_2.as_ref().unwrap();

        for j in 0..n_features {
            let s = mat.indptr[j];
            let e = mat.indptr[j + 1];
            let col_indices = &mat.indices[s..e];
            let col_vals = &vals[s..e];

            let mut min_v = 0_f32;
            let mut max_v = 0_f32;
            for &v in col_vals {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let range = max_v - min_v;
            mins.push(min_v);
            ranges.push(range);

            let offset = j * n_cells;

            if range > 1e-10 {
                let scale = 255.0 / range;
                for i in 0..col_indices.len() {
                    let cell_idx = col_indices[i];
                    let val = col_vals[i];
                    let q_val = ((val - min_v) * scale).round() as u8;
                    data[offset + cell_idx] = q_val;
                }
            }
        }

        Self {
            data,
            n_cells,
            n_features,
            feature_min: mins,
            feature_range: ranges,
        }
    }

    #[inline(always)]
    pub fn get_col(&self, tf_idx: usize) -> &[u8] {
        let start = tf_idx * self.n_cells;
        &self.data[start..start + self.n_cells]
    }
}

////////////////
// Histograms //
////////////////

#[derive(Clone, Copy, Default)]
struct HistogramBin {
    count: usize,
    y_sum: f32,
    y_sum_sq: f32,
}

//////////////////
// Tree helpers //
//////////////////

#[allow(dead_code)]
enum Node {
    Leaf {
        mean: f32,
    },
    Split {
        feature_idx: usize,
        threshold: f32,
        left: usize,
        right: usize,
        weighted_impurity_decrease: f32,
    },
}

#[inline]
fn node_variance(sum: f32, sum_sq: f32, n: usize) -> f32 {
    if n < 2 {
        return 0_f32;
    }
    let nf = n as f32;
    f32::max(0_f32, sum_sq / nf - (sum / nf) * (sum / nf))
}

///////////////////
// Tree building //
///////////////////

struct TreeBuffers {
    feat_buf: Vec<usize>,
    left_buf: Vec<u32>,
    right_buf: Vec<u32>,
    left_y_buf: Vec<f32>,  // NEW: packed Y array
    right_y_buf: Vec<f32>, // NEW: packed Y array
    hist: [HistogramBin; 256],
    cum_hist: [HistogramBin; 256],
}

impl TreeBuffers {
    fn new(n_features: usize, n_samples: usize) -> Self {
        Self {
            feat_buf: (0..n_features).collect(),
            left_buf: vec![0; n_samples],
            right_buf: vec![0; n_samples],
            left_y_buf: vec![0.0; n_samples],
            right_y_buf: vec![0.0; n_samples],
            hist: [HistogramBin::default(); 256],
            cum_hist: [HistogramBin::default(); 256],
        }
    }

    #[inline]
    fn build_histograms(&mut self, tf_col: &[u8], sample_slice: &[u32], y_slice: &[f32]) {
        self.hist.fill(HistogramBin::default());

        // Perfect sequential iteration, zero cache misses!
        for i in 0..sample_slice.len() {
            let bin_idx = tf_col[sample_slice[i] as usize] as usize;
            let y = y_slice[i];
            self.hist[bin_idx].count += 1;
            self.hist[bin_idx].y_sum += y;
            self.hist[bin_idx].y_sum_sq += y * y;
        }

        let mut acc = HistogramBin::default();
        for i in 0..256 {
            acc.count += self.hist[i].count;
            acc.y_sum += self.hist[i].y_sum;
            acc.y_sum_sq += self.hist[i].y_sum_sq;
            self.cum_hist[i] = acc;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_split(
    threshold: usize,
    feat: usize,
    parent_var: f32,
    n: usize,
    y_sum: f32,
    y_sum_sq: f32,
    bufs: &TreeBuffers,
    config: &dyn TreeRegressorConfig,
    best_score: &mut f32,
    best_feature: &mut usize,
    best_threshold_u8: &mut u8,
    best_y_sum_l: &mut f32,
    best_y_sum_sq_l: &mut f32,
) {
    let left_stats = &bufs.cum_hist[threshold];
    let n_left = left_stats.count;
    let n_right = n - n_left;

    if n_left < config.min_samples_leaf() || n_right < config.min_samples_leaf() {
        return;
    }

    let y_sum_l = left_stats.y_sum;
    let y_sum_sq_l = left_stats.y_sum_sq;
    let y_sum_r = y_sum - y_sum_l;
    let y_sum_sq_r = y_sum_sq - y_sum_sq_l;

    let score = parent_var
        - (n_left as f32 / n as f32) * node_variance(y_sum_l, y_sum_sq_l, n_left)
        - (n_right as f32 / n as f32) * node_variance(y_sum_r, y_sum_sq_r, n_right);

    if score > *best_score {
        *best_score = score;
        *best_feature = feat;
        *best_threshold_u8 = threshold as u8;
        *best_y_sum_l = y_sum_l;
        *best_y_sum_sq_l = y_sum_sq_l;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    y_slice: &mut [f32], // NEW: Packed Y array!
    x: &DenseQuantisedStore,
    sample_slice: &mut [u32],
    y_sum: f32,
    y_sum_sq: f32,
    n_total: usize,
    n_features_split: usize,
    config: &dyn TreeRegressorConfig,
    depth: usize,
    nodes: &mut Vec<Node>,
    bufs: &mut TreeBuffers,
    rng: &mut SmallRng,
) -> usize {
    let n = sample_slice.len();
    let mean = y_sum / n as f32;
    let parent_var = node_variance(y_sum, y_sum_sq, n);

    let max_depth_reached = config.max_depth().map_or(false, |d| depth >= d);

    if n < 2 * config.min_samples_leaf() || parent_var < config.min_variance() || max_depth_reached
    {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    let n_features = x.n_features;
    let k = n_features_split.min(n_features);

    for i in 0..k {
        let j = rng.random_range(i..n_features);
        bufs.feat_buf.swap(i, j);
    }

    let mut best_score = 0.0f32;
    let mut best_feature = usize::MAX;
    let mut best_threshold_u8 = 0u8;
    let mut best_y_sum_l = 0.0f32;
    let mut best_y_sum_sq_l = 0.0f32;

    for fi_idx in 0..k {
        let feat = bufs.feat_buf[fi_idx];
        let tf_col = x.get_col(feat);

        bufs.build_histograms(tf_col, sample_slice, y_slice);

        let min_bin = bufs.hist.iter().position(|b| b.count > 0).unwrap_or(0);
        let max_bin = bufs.hist.iter().rposition(|b| b.count > 0).unwrap_or(255);

        if min_bin == max_bin {
            continue;
        }

        if config.random_threshold() {
            for _ in 0..config.n_thresholds() {
                let threshold = rng.random_range(min_bin..max_bin);
                evaluate_split(
                    threshold,
                    feat,
                    parent_var,
                    n,
                    y_sum,
                    y_sum_sq,
                    bufs,
                    config,
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_y_sum_l,
                    &mut best_y_sum_sq_l,
                );
            }
        } else {
            for threshold in min_bin..max_bin {
                evaluate_split(
                    threshold,
                    feat,
                    parent_var,
                    n,
                    y_sum,
                    y_sum_sq,
                    bufs,
                    config,
                    &mut best_score,
                    &mut best_feature,
                    &mut best_threshold_u8,
                    &mut best_y_sum_l,
                    &mut best_y_sum_sq_l,
                );
            }
        }
    }

    if best_feature == usize::MAX {
        let idx = nodes.len();
        nodes.push(Node::Leaf { mean });
        return idx;
    }

    let tf_col = x.get_col(best_feature);
    let mut l_idx = 0;
    let mut r_idx = 0;

    for i in 0..n {
        let s = sample_slice[i];
        let y = y_slice[i]; // Fetch aligned Y
        let val = tf_col[s as usize];

        let is_right = (val > best_threshold_u8) as usize;
        let is_left = 1 - is_right;

        // Partition BOTH arrays perfectly seamlessly
        bufs.left_buf[l_idx] = s;
        bufs.left_y_buf[l_idx] = y;
        bufs.right_buf[r_idx] = s;
        bufs.right_y_buf[r_idx] = y;

        l_idx += is_left;
        r_idx += is_right;
    }

    sample_slice[..l_idx].copy_from_slice(&bufs.left_buf[..l_idx]);
    sample_slice[l_idx..].copy_from_slice(&bufs.right_buf[..r_idx]);

    y_slice[..l_idx].copy_from_slice(&bufs.left_y_buf[..l_idx]);
    y_slice[l_idx..].copy_from_slice(&bufs.right_y_buf[..r_idx]);

    let y_sum_r = y_sum - best_y_sum_l;
    let y_sum_sq_r = y_sum_sq - best_y_sum_sq_l;

    let node_idx = nodes.len();
    nodes.push(Node::Split {
        feature_idx: best_feature,
        threshold: x.feature_min[best_feature]
            + (best_threshold_u8 as f32 / 255.0) * x.feature_range[best_feature],
        left: usize::MAX,
        right: usize::MAX,
        weighted_impurity_decrease: (n as f32 / n_total as f32) * best_score,
    });

    let (left_sl, right_sl) = sample_slice.split_at_mut(l_idx);
    let (left_y_sl, right_y_sl) = y_slice.split_at_mut(l_idx);

    let left_idx = build_node(
        left_y_sl,
        x,
        left_sl,
        best_y_sum_l,
        best_y_sum_sq_l,
        n_total,
        n_features_split,
        config,
        depth + 1,
        nodes,
        bufs,
        rng,
    );
    let right_idx = build_node(
        right_y_sl,
        x,
        right_sl,
        y_sum_r,
        y_sum_sq_r,
        n_total,
        n_features_split,
        config,
        depth + 1,
        nodes,
        bufs,
        rng,
    );

    if let Node::Split { left, right, .. } = &mut nodes[node_idx] {
        *left = left_idx;
        *right = right_idx;
    }

    node_idx
}

fn accumulate_importances(nodes: &[Node], importances: &mut [f32]) {
    for node in nodes {
        if let Node::Split {
            feature_idx,
            weighted_impurity_decrease,
            ..
        } = node
        {
            importances[*feature_idx] += weighted_impurity_decrease;
        }
    }
}

fn build_y_dense(target_variable: &SparseAxis<u16, f32>, n_samples: usize) -> Vec<f32> {
    let (y_indices, y_data) = target_variable.get_indices_data_2();
    let mut y_dense = vec![0.0f32; n_samples];
    for (i, &idx) in y_indices.iter().enumerate() {
        y_dense[idx] = y_data[i];
    }
    y_dense
}

fn fit_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &dyn TreeRegressorConfig,
    seed: usize,
) -> Vec<f32> {
    let n_features = feature_matrix.n_features;
    let n_features_split = if config.n_features_split() == 0 {
        ((n_features as f64).sqrt() as usize).max(1)
    } else {
        config.n_features_split()
    };

    let n_sub = if config.subsample_rate() >= 1.0 {
        n_samples
    } else {
        ((n_samples as f32 * config.subsample_rate()).round() as usize)
            .max(2 * config.min_samples_leaf())
    };

    let y_dense = build_y_dense(target_variable, n_samples);

    let mut sample_indices: Vec<u32> = vec![0; n_samples];
    let mut root_y_buf: Vec<f32> = vec![0.0; n_samples]; // NEW: Independent buffer
    let mut bufs = TreeBuffers::new(n_features, n_samples);
    let mut nodes: Vec<Node> = Vec::new();
    let mut importances = vec![0.0f32; n_features];

    for tree_idx in 0..config.n_trees() {
        nodes.clear();
        let mut rng =
            SmallRng::seed_from_u64(seed.wrapping_add(tree_idx * 6364136223846793005) as u64);

        let active_len = if n_sub < n_samples {
            if config.bootstrap() {
                // Correct Bootstrapping logic: duplicates are expected!
                for i in 0..n_sub {
                    sample_indices[i] = rng.random_range(0..n_samples as u32);
                }
                n_sub
            } else {
                sample_indices
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = i as u32);
                for i in 0..n_sub {
                    let j = rng.random_range(i..n_samples);
                    sample_indices.swap(i, j);
                }
                n_sub
            }
        } else {
            sample_indices
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = i as u32);
            n_samples
        };

        let active = &mut sample_indices[..active_len];
        let root_y = &mut root_y_buf[..active_len];

        // Pack Y directly next to indices
        let mut y_sum = 0.0f32;
        let mut y_sum_sq = 0.0f32;
        for i in 0..active_len {
            let s = active[i] as usize;
            let y = y_dense[s];
            root_y[i] = y;
            y_sum += y;
            y_sum_sq += y * y;
        }

        build_node(
            root_y,
            feature_matrix,
            active,
            y_sum,
            y_sum_sq,
            active_len,
            n_features_split,
            config,
            0,
            &mut nodes,
            &mut bufs,
            &mut rng,
        );

        accumulate_importances(&nodes, &mut importances);
    }

    let total: f32 = sum_simd_f32(&importances);
    if total > 0.0 {
        importances.iter_mut().for_each(|v| *v /= total);
    }
    importances
}

fn fit_extra_trees(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &ExtraTreesConfig,
    seed: usize,
) -> Vec<f32> {
    fit_trees(target_variable, feature_matrix, n_samples, config, seed)
}

fn fit_random_forest(
    target_variable: &SparseAxis<u16, f32>,
    feature_matrix: &DenseQuantisedStore,
    n_samples: usize,
    config: &RandomForestConfig,
    seed: usize,
) -> Vec<f32> {
    fit_trees(target_variable, feature_matrix, n_samples, config, seed)
}

//////////
// Main //
//////////

pub fn scenic_gene_filter(
    f_path: &str,
    cell_indices: &[usize],
    min_counts: usize,
    min_cells: f32,
) -> Vec<usize> {
    let reader = ParallelSparseReader::new(f_path).unwrap();
    let total_genes = reader.get_header().total_genes;
    let all_gene_indices: Vec<usize> = (0..total_genes).collect();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let n_cells = cell_indices.len();

    let mut passing = Vec::new();

    for chunk in all_gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE) {
        let mut gene_chunks = reader.read_gene_parallel(chunk);
        gene_chunks.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&cell_set);
        });

        for gene in &gene_chunks {
            let total_counts: u32 = gene.data_raw.iter().map(|&x| x as u32).sum();
            let expressed_fraction = gene.nnz as f32 / n_cells as f32;

            if total_counts >= min_counts as u32 && expressed_fraction >= min_cells {
                passing.push(gene.original_index);
            }
        }
    }

    passing
}

pub fn run_scenic_grn(
    f_path: &str,
    cell_indices: &[usize],
    gene_indices: &[usize],
    tf_indices: &[usize],
    learner: &RegressionLearner,
    seed: usize,
    verbose: bool,
) -> Mat<f32> {
    let start_total = Instant::now();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();
    let start_reading = Instant::now();
    let reader = ParallelSparseReader::new(f_path).unwrap();

    let mut gene_chunks: Vec<CscGeneChunk> = reader.read_gene_parallel(tf_indices);
    gene_chunks.par_iter_mut().for_each(|chunk| {
        chunk.filter_selected_cells(&cell_set);
    });

    let end_reading = start_reading.elapsed();
    let tf_data: CompressedSparseData<u16, f32> =
        from_gene_chunks::<u16>(&gene_chunks, cell_set.len());
    let tf_data = DenseQuantisedStore::from_csc(&tf_data, cell_set.len());

    if verbose {
        println!(
            "Loaded in and filtered TF data (n: {}) to cells of interest: {:.2?}",
            tf_data.n_features, end_reading
        );
    }

    let mut importance_scores: Vec<Vec<f32>> = vec![Vec::new(); gene_indices.len()];

    for (chunk_idx, chunk) in gene_indices.chunks(SCENIC_GENE_CHUNK_SIZE).enumerate() {
        if verbose {
            println!(
                "Processing gene chunk {}/{} ({} genes)",
                chunk_idx + 1,
                gene_indices.len().div_ceil(SCENIC_GENE_CHUNK_SIZE),
                chunk.len()
            );
        }

        let start_chunk = Instant::now();
        let mut gene_chunks_target: Vec<CscGeneChunk> = reader.read_gene_parallel(chunk);
        gene_chunks_target.par_iter_mut().for_each(|c| {
            c.filter_selected_cells(&cell_set);
        });

        let sparse_columns: Vec<SparseAxis<u16, f32>> = gene_chunks_target
            .iter()
            .map(|c| c.to_sparse_axis(cell_set.len()))
            .collect();

        let chunk_importances: Vec<Vec<f32>> = sparse_columns
            .par_iter()
            .map(|gene| match learner {
                RegressionLearner::ExtraTrees(cfg) => {
                    fit_trees(gene, &tf_data, cell_set.len(), cfg, seed)
                }
                RegressionLearner::RandomForest(cfg) => {
                    fit_trees(gene, &tf_data, cell_set.len(), cfg, seed)
                }
            })
            .collect();

        let base = chunk_idx * SCENIC_GENE_CHUNK_SIZE;
        for (i, importances) in chunk_importances.into_iter().enumerate() {
            importance_scores[base + i] = importances;
        }

        if verbose {
            println!("  Chunk done in {:.2?}", start_chunk.elapsed());
        }
    }

    if verbose {
        println!(
            "SCENIC GRN inference complete in {:.2?}",
            start_total.elapsed()
        );
    }

    let n_genes = importance_scores.len();
    let n_tfs = if n_genes > 0 {
        importance_scores[0].len()
    } else {
        0
    };

    Mat::from_fn(n_genes, n_tfs, |i, j| importance_scores[i][j])
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_variance_basic() {
        let (sum, sum_sq) = (6.0f32, 14.0f32);
        let v = node_variance(sum, sum_sq, 3);
        assert!((v - 2.0 / 3.0).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn node_variance_uniform() {
        let (sum, sum_sq) = (9.0f32, 27.0f32);
        let v = node_variance(sum, sum_sq, 3);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn branchless_partitioning_logic_packed() {
        let tf_col: Vec<u8> = vec![10, 50, 200, 30, 250, 100];
        let sample_slice: Vec<u32> = vec![0, 1, 2, 4];
        let mut y_slice: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut left_buf = vec![0; 4];
        let mut right_buf = vec![0; 4];
        let mut left_y_buf = vec![0.0; 4];
        let mut right_y_buf = vec![0.0; 4];

        let threshold = 100u8;
        let mut l_idx = 0;
        let mut r_idx = 0;

        for i in 0..sample_slice.len() {
            let s = sample_slice[i];
            let y = y_slice[i];
            let val = tf_col[s as usize];
            let is_right = (val > threshold) as usize;
            let is_left = 1 - is_right;

            left_buf[l_idx] = s;
            left_y_buf[l_idx] = y;
            right_buf[r_idx] = s;
            right_y_buf[r_idx] = y;

            l_idx += is_left;
            r_idx += is_right;
        }

        assert_eq!(l_idx, 2);
        assert_eq!(r_idx, 2);

        assert_eq!(&left_buf[..l_idx], &[0, 1]);
        assert_eq!(&right_buf[..r_idx], &[2, 4]);

        // Assert packed Y array partitioned perfectly along with it
        assert_eq!(&left_y_buf[..l_idx], &[1.0, 2.0]);
        assert_eq!(&right_y_buf[..r_idx], &[3.0, 4.0]);
    }
}
