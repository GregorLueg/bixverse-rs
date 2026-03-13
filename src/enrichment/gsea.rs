//! Gene set enrichment analysis based on the work of Korotkevich, et al.,
//! bioRxiv, 2021.

use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use statrs::distribution::{Beta, ContinuousCDF};
use statrs::function::gamma::digamma;
use std::ops::{Add, AddAssign};

use crate::core::math::stats::trigamma;
use crate::prelude::*;

//////////////////
// Type aliases //
//////////////////

/// Type alias for multi level error calculations.
///
/// ### Fields
///
/// * `0` - Simple error for the multi level fgsea
/// * `1` - Multi error for the multi level fgsea
pub type MultiLevelErrRes<T> = (Vec<T>, Vec<T>);

////////////////
// Structures //
////////////////

/////////////
// Results //
/////////////

/// Structure to store GSEA stats
#[derive(Clone, Debug)]
pub struct GseaStats<T> {
    /// Enrichment score
    pub es: T,
    /// The index positions of the leading edge genes
    pub leading_edge: Vec<i32>,
    /// Top points for plotting purposes
    pub top: Vec<T>,
    /// Bottom points for plotting purposes
    pub bottom: Vec<T>,
}

/// Structure for final GSEA results from any algorithm
#[derive(Clone, Debug)]
pub struct GseaResults<'a, T> {
    /// Enrichment score
    pub es: &'a [T],
    /// Normalised enrichment score
    pub nes: Vec<Option<T>>,
    /// The p-value for this pathway based on permutations
    pub pvals: Vec<T>,
    /// The number of permutations with higher/lower ES
    pub n_more_extreme: Vec<usize>,
    /// The number of permutation ES ≤ 0
    pub le_zero: Vec<usize>,
    /// The number of permutation ES ≥ 0
    pub ge_zero: Vec<usize>,
    /// The size of the pathway
    pub size: &'a [usize],
}

/// Structure for results from the different GSEA permutation methods
#[derive(Clone, Debug)]
pub struct GseaBatchResults<T> {
    /// The number of permutations less than the score
    pub le_es: Vec<usize>,
    /// The number of permutations greater than the score
    pub ge_es: Vec<usize>,
    /// The number of permutation ES ≤ 0
    pub le_zero: Vec<usize>,
    /// The number of permutation ES ≥ 0
    pub ge_zero: Vec<usize>,
    /// The sum of permuted enrichment scores that were ≤ 0
    pub le_zero_sum: Vec<T>,
    /// The sum of permuted enrichment scores that were ≥ 0
    pub ge_zero_sum: Vec<T>,
}

////////////
// Params //
////////////

/// Structure to store GSEA params
#[derive(Clone, Debug)]
pub struct GseaParams<T> {
    /// The GSEA parameter, usually 1.0
    pub gsea_param: T,
    /// The maximum size of the allowed pathways
    pub max_size: usize,
    /// The minimum size of the allowed pathways
    pub min_size: usize,
}

/////////////
// Helpers //
/////////////

/// Calculate the enrichment score (based on the fgsea C++ implementation)
///
/// ### Params
///
/// * `ranks` - Gene ranks array
/// * `pathway_indices` - Indices of genes in the pathway
///
/// ### Returns
///
/// Enrichment score value
fn calc_es<T: BixverseFloat>(ranks: &[T], pathway_indices: &[usize]) -> T {
    let n = ranks.len();
    let k = pathway_indices.len();

    // fast path for empty or saturated pathways
    if k == 0 || k == n {
        return T::zero();
    }

    let mut ns = T::zero();
    for &p in pathway_indices {
        ns += ranks[p];
    }

    let q1 = T::one() / T::from_usize(n - k).unwrap();
    let q2 = T::one() / ns;

    let mut res = T::zero();
    let mut res_abs = T::zero();
    let mut cur = T::zero();

    // track the next expected index to calculate gaps using purely unsigned
    // math
    let mut next_expected = 0;

    for &p in pathway_indices {
        let gap = p - next_expected;

        // only apply penalty and check max if there was actually a gap
        // (misses)
        if gap > 0 {
            cur -= q1 * T::from_usize(gap).unwrap();
            let cur_abs = cur.abs();
            if cur_abs > res_abs {
                res = cur;
                res_abs = cur_abs;
            }
        }

        // Apply reward (hit)
        cur += q2 * ranks[p];
        let cur_abs = cur.abs();
        if cur_abs > res_abs {
            res = cur;
            res_abs = cur_abs;
        }

        next_expected = p + 1;
    }

    res
}

/// Calculate the positive enrichment score
///
/// Based on the fgsea C++ implementation
///
/// ### Params
///
/// * `ranks` - Gene ranks array
/// * `pathway_indices` - Indices of genes in the pathway
///
/// # Returns
///
/// Positive enrichment score value
fn calc_positive_es<T: BixverseFloat>(ranks: &[T], pathway_indices: &[usize]) -> T {
    let n = ranks.len();
    let k = pathway_indices.len();

    if k == 0 || k == n {
        return T::zero();
    }

    let mut ns = T::zero();
    for &p in pathway_indices {
        ns += ranks[p];
    }

    let q1 = T::one() / T::from_usize(n - k).unwrap();
    let q2 = T::one() / ns;

    let mut res = T::zero();
    let mut cur = T::zero();
    let mut next_expected = 0;

    for &p in pathway_indices {
        let gap = p - next_expected;

        if gap > 0 {
            cur -= q1 * T::from_usize(gap).unwrap();
        }
        cur += q2 * ranks[p];

        // A simple branch generally compiles down to a fast `cmov` instruction,
        // outperforming float-specific `.max()` trait boundaries in hot loops.
        if cur > res {
            res = cur;
        }

        next_expected = p + 1;
    }

    res
}

/// Generate k random numbers from [a, b] inclusive range using Fisher-Yates shuffle
///
/// ### Params
///
/// * `a` - Range start
/// * `b` - Range end
/// * `k` - Number of elements to select
/// * `rng` - Random number generator
///
/// ### Returns
///
/// Sorted vector of k random elements
///
/// ### Panics
///
/// If k > range size (b - a + 1)
fn combination(a: usize, b: usize, k: usize, rng: &mut impl Rng) -> Vec<usize> {
    let n = b - a + 1;
    if k > n {
        panic!("k cannot be greater than range size n");
    }

    let mut indices: Vec<usize> = (a..=b).collect();

    for i in 0..k {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }

    let mut result: Vec<usize> = indices.into_iter().take(k).collect();
    result.sort_unstable();

    result
}

/// Helper to calculate the beta mean log
///
/// ### Params
///
/// * `a` - First beta parameter
/// * `b` - Second beta parameter
///
/// ### Returns
///
/// Beta mean log value
fn beta_mean_log(a: usize, b: usize) -> f64 {
    digamma(a as f64) - digamma((b + 1) as f64)
}

/// Rearranges array so nth element is in its sorted position
///
/// ### Params
///
/// * `arr` - Array to rearrange
/// * `n` - Target position
fn nth_element(arr: &mut [i32], n: usize) {
    if arr.is_empty() || n >= arr.len() {
        return;
    }
    arr.select_nth_unstable(n);
}

/// Calculates the log corrections for p-value adjustment
///
/// ### Params
///
/// * `prob_corrector` - Probability correction vector
/// * `prob_corr_idx` - Correction index
/// * `sample_size` - Sample size
///
/// ### Returns
///
/// Tuple of (log correction, validity flag)
fn calc_log_correction(
    prob_corrector: &[usize],
    prob_corr_idx: usize,
    sample_size: usize,
) -> (f64, bool) {
    let mut result = 0.0;
    #[allow(clippy::manual_div_ceil)]
    let half_size = (sample_size + 1) / 2;
    let remainder = sample_size - (prob_corr_idx % half_size);

    let cond_prob = beta_mean_log(prob_corrector[prob_corr_idx] + 1, remainder);
    result += cond_prob;

    if cond_prob.exp() >= 0.5 {
        (result, true)
    } else {
        (result, false)
    }
}

/// Convert order indices to ranks (from fgsea C++ implementation)
///
/// ### Params
///
/// * `order` - Ordering indices
///
/// ### Returns
///
/// Rank vector
pub fn ranks_from_order(order: &[usize]) -> Vec<i32> {
    let mut res = vec![0; order.len()];
    for (i, _) in order.iter().enumerate() {
        let idx = order[i];
        res[idx] = i as i32;
    }
    res
}

/// Returns a vector of indices that would sort the input slice
/// Implementation of the C++ code
///
/// ### Params
///
/// * `x` - Input slice to get ordering for
///
/// ### Returns
///
/// Vector of sorting indice
pub fn fgsea_order<T>(x: &[T]) -> Vec<usize>
where
    T: PartialOrd,
{
    let mut res: Vec<usize> = (0..x.len()).collect();
    res.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    res
}

/// Extract values at specified indices from a vector
///
/// ### Params
///
/// * `from` - Source vector
/// * `indices` - Indices to extract (1-indexed)
///
/// # Returns
///
/// Option containing extracted values or None if invalid index
fn subvector<T: BixverseFloat>(from: &[T], indices: &[usize]) -> Option<Vec<T>> {
    let mut result = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx > 0 && idx <= from.len() {
            result.push(from[idx - 1]);
        } else {
            return None;
        }
    }
    Some(result)
}

/// Generate random gene set indices using parallel processing
///
/// ### Params
///
/// * `iter_number` - Number of iterations/permutations
/// * `max_len` - Maximum length of each sample
/// * `universe_length` - Total number of genes
/// * `seed` - Random seed
/// * `one_indexed` - Whether to use 1-based indexing
///
/// # Returns
///
/// Vector of random index vectors
pub fn create_random_gs_indices(
    iter_number: usize,
    max_len: usize,
    universe_length: usize,
    seed: u64,
    one_indexed: bool,
) -> Vec<Vec<usize>> {
    (0..iter_number)
        .into_par_iter()
        .map(|i| {
            let iter_seed = seed.wrapping_add(i as u64);
            let mut rng = StdRng::seed_from_u64(iter_seed);

            let adjusted_universe = universe_length - 1;
            let actual_len = std::cmp::min(max_len, adjusted_universe);

            let mut indices: Vec<usize> = (0..adjusted_universe).collect();

            for i in 0..actual_len {
                let j = rng.random_range(i..indices.len());
                indices.swap(i, j);
            }

            indices.truncate(actual_len);

            if one_indexed {
                indices.iter_mut().for_each(|x| *x += 1);
            }

            indices
        })
        .collect()
}

/// Calculate the ES and leading edge genes
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `gs_idx` - Gene set indices
/// * `gsea_param` - GSEA parameter for weighting
/// * `return_leading_edge` - Whether to return leading edge genes
/// * `return_all_extreme` - Whether to return the all points for plotting
/// * `one_indexed` - Whether indices are one-based
///
/// ### Returns
///
/// Tuple of (gene statistic, leading edge genes)
pub fn calc_gsea_stats<T>(
    stats: &[T],
    gs_idx: &[i32],
    gsea_param: T,
    return_leading_edge: bool,
    return_all_extreme: bool,
    one_indexed: bool,
) -> GseaStats<T>
where
    T: BixverseFloat,
{
    let n = stats.len();
    let m = gs_idx.len();
    let mut r_adj = Vec::with_capacity(m);
    for &i in gs_idx {
        let idx = if one_indexed { i - 1 } else { i } as usize;
        r_adj.push(stats[idx].abs().powf(gsea_param));
    }
    let nr: T = r_adj.iter().copied().fold(T::zero(), |acc, x| acc + x);
    let r_cum_sum: Vec<T> = if nr == T::zero() {
        gs_idx
            .iter()
            .enumerate()
            .map(|(i, _)| T::from_usize(i).unwrap() / T::from_usize(r_adj.len()).unwrap())
            .collect()
    } else {
        cumsum(&r_adj).iter().map(|x| *x / nr).collect()
    };
    let n_t = T::from_usize(n).unwrap();
    let m_t = T::from_usize(m).unwrap();
    let top_tmp: Vec<T> = gs_idx
        .iter()
        .enumerate()
        .map(|(i, x)| (T::from_i32(*x).unwrap() - T::from_usize(i + 1).unwrap()) / (n_t - m_t))
        .collect();
    let tops: Vec<T> = r_cum_sum
        .iter()
        .zip(top_tmp.iter())
        .map(|(x1, x2)| *x1 - *x2)
        .collect();
    let bottoms: Vec<T> = if nr == T::zero() {
        tops.iter().map(|x| *x - (T::one() / m_t)).collect()
    } else {
        tops.iter()
            .zip(r_adj.iter())
            .map(|(top, adj)| *top - (*adj / nr))
            .collect()
    };
    let max_p = array_max(&tops);
    let min_p = array_min(&bottoms);
    let gene_stat = if max_p == -min_p {
        T::zero()
    } else if max_p > -min_p {
        max_p
    } else {
        min_p
    };
    let leading_edge = if return_leading_edge {
        if max_p > -min_p {
            let max_idx = bottoms
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            gs_idx.iter().take(max_idx + 1).cloned().collect()
        } else if max_p < -min_p {
            let min_idx = bottoms
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            gs_idx.iter().skip(min_idx).cloned().rev().collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    if return_all_extreme {
        GseaStats {
            es: gene_stat,
            leading_edge,
            top: tops,
            bottom: bottoms,
        }
    } else {
        GseaStats {
            es: gene_stat,
            leading_edge,
            top: Vec::new(),
            bottom: Vec::new(),
        }
    }
}

//////////////////
// Segment tree //
//////////////////

/// Structure from the fgsea simple algorithm implementing a segment tree data structure
///
/// ### Fields
///
/// * `t` - Tree array for storing segment values
/// * `b` - Block array for storing block-level aggregates
/// * `n` - Total size of the tree (padded to power of 2)
/// * `k2` - Number of blocks in the structure
/// * `log_k` - Log base 2 of block size for bit operations
/// * `block_mask` - Bitmask for extracting block positions
#[derive(Clone, Debug)]
pub struct SegmentTree<T> {
    t: Vec<T>,
    b: Vec<T>,
    n: usize,
    k2: usize,
    log_k: usize,
    block_mask: usize,
}

impl<T> SegmentTree<T>
where
    T: Copy + AddAssign + Default + Add<Output = T>,
{
    /// Create a new segment tree with size n
    ///
    /// ### Params
    ///
    /// * `n_` - Size of the tree
    ///
    /// ### Returns
    ///
    /// Initialised structure
    pub fn new(n_: usize) -> Self {
        let mut k = 1;
        let mut log_k = 0;

        while k * k < n_ {
            k <<= 1;
            log_k += 1;
        }

        let k2 = (n_ - 1) / k + 1;
        let n = k * k;
        let block_mask = k - 1;
        let t = vec![T::default(); n];
        let b = vec![T::default(); k2];

        SegmentTree {
            t,
            b,
            n,
            k2,
            log_k,
            block_mask,
        }
    }

    /// Increment the value at position p by delta
    ///
    /// ### Params
    ///
    /// * `p` - Position to increment (mutable)
    /// * `delta` - Value to add
    ///
    /// ### Note
    ///
    /// p NEEDS to be mutable for the algorithm to work correctly
    pub fn increment(&mut self, mut p: usize, delta: T) {
        let block_end = p - (p & self.block_mask) + self.block_mask + 1;
        while p < block_end && p < self.n {
            self.t[p] += delta;
            p += 1;
        }
        let mut p1 = p >> self.log_k;
        while p1 < self.k2 {
            self.b[p1] += delta;
            p1 += 1;
        }
    }

    /// Calculate the sum in range 0 to r
    ///
    /// ### Params
    ///
    /// * `r` - Right bound of range
    ///
    /// ### Returns
    ///
    /// Sum in range [0, r)
    pub fn query_r(&self, mut r: usize) -> T {
        if r == 0 {
            return T::default();
        }
        r -= 1;
        self.t[r] + self.b[r >> self.log_k]
    }
}

///////////////////
// Sample Chunks //
///////////////////

/// Structure for fgsea multi-level
/// Stores and manages chunked samples for efficient processing
///
/// ### Fields
///
/// * `chunk_sum` - Sum of rank values in each chunk for fast computation
/// * `chunks` - Vector of chunks, each containing gene indices for that chunk
#[derive(Clone, Debug)]
struct SampleChunks<T> {
    chunk_sum: Vec<T>,
    chunks: Vec<Vec<i32>>,
}

impl<T: BixverseFloat> SampleChunks<T> {
    /// Creates new SampleChunks with specified number of chunks
    ///
    /// ### Params
    ///
    /// * `chunks_number` - Number of chunks to create
    ///
    /// ### Returns
    ///
    /// Initialised structure
    fn new(chunks_number: usize) -> Self {
        Self {
            chunk_sum: vec![T::zero(); chunks_number],
            chunks: vec![Vec::new(); chunks_number],
        }
    }
}

//////////////
// ES Ruler //
//////////////

//////////////
// ES Ruler //
//////////////

/// This is EsRuler implementation for adaptive enrichment score sampling
///
/// Further structure for fgsea multi-level. This has been memory-optimized
/// by flattening the nested vectors to improve CPU cache locality while
/// preserving exact parity with the original C++ algorithm's MCMC steps.
#[derive(Clone, Debug)]
struct EsRuler<T> {
    /// Gene ranks used for ES calculation.
    ranks: Vec<T>,
    /// Current sample size (may change slightly during duplication).
    sample_size: usize,
    /// Original sample size for p-value calculation.
    original_sample_size: usize,
    /// Number of genes in the pathway.
    pathway_size: usize,
    /// Flattened 1D vector storing the current sample sets being processed.
    /// To access sample `i`, we slice
    /// `[i * pathway_size .. (i+1) * pathway_size]`.
    current_samples: Vec<usize>,
    /// Calculated enrichment scores from samples.
    enrichment_scores: Vec<T>,
    /// Probability correction factors for p-value adjustment.
    prob_corrector: Vec<usize>,
    /// Number of chunks used for optimisation.
    chunks_number: i32,
    /// Last element index in each chunk.
    chunk_last_element: Vec<i32>,
    /// Buffer to hold the next generation of samples without reallocating
    next_samples: Vec<usize>,
    /// Reusable buffers for sorting and tracking stats in duplicate_samples
    stats_buf: Vec<(T, usize)>,
    /// Is positive ES buffer
    is_pos_es_buf: Vec<bool>,
}

impl<T: BixverseFloat> EsRuler<T> {
    /// Creates a new ES ruler for adaptive sampling
    ///
    /// ### Params
    ///
    /// * `inp_ranks` - Input gene ranks
    /// * `inp_sample_size` - Sample size
    /// * `inp_pathway_size` - Pathway size
    ///
    /// ### Returns
    ///
    /// Initialised structure
    fn new(inp_ranks: &[T], inp_sample_size: usize, inp_pathway_size: usize) -> Self {
        Self {
            ranks: inp_ranks.to_vec(),
            sample_size: inp_sample_size,
            original_sample_size: inp_sample_size,
            pathway_size: inp_pathway_size,
            // Pre-allocate the flattened array filled with 0s
            current_samples: vec![0; inp_sample_size * inp_pathway_size],
            enrichment_scores: Vec::new(),
            prob_corrector: Vec::new(),
            chunks_number: 0,
            chunk_last_element: Vec::new(),
            // pre-allocate buffers
            next_samples: Vec::with_capacity(inp_sample_size * inp_pathway_size),
            stats_buf: vec![(T::zero(), 0); inp_sample_size],
            is_pos_es_buf: vec![false; inp_sample_size],
        }
    }

    /// Helper to get a read-only slice for a specific sample from the flattened
    /// array
    ///
    /// ### Params
    ///
    /// * `index` - Index position of pathway to return
    ///
    /// ### Returns
    ///
    /// The slice of pathway positions for the genes
    #[inline(always)]
    fn get_sample(&self, idx: usize) -> &[usize] {
        let start = idx * self.pathway_size;
        &self.current_samples[start..start + self.pathway_size]
    }

    /// Helper to get a mutable slice for a specific sample from the flattened
    /// array
    ///
    /// ### Params
    ///
    /// * `index` - Index position of pathway to return
    ///
    /// ### Returns
    ///
    /// A mutable slice of pathway positions for the genes
    #[inline(always)]
    fn get_sample_mut(&mut self, idx: usize) -> &mut [usize] {
        let start = idx * self.pathway_size;
        &mut self.current_samples[start..start + self.pathway_size]
    }

    /// Removes samples with low ES and duplicates samples with high ES
    ///
    /// This drives the sampling process toward higher and higher ES values.
    /// Uses a boolean mask instead of FxHashSet for faster positive ES
    /// tracking.
    fn duplicate_samples(&mut self) {
        self.is_pos_es_buf[..self.sample_size].fill(false);
        let mut total_pos_es_count: i32 = 0;

        for sample_id in 0..self.sample_size {
            // Inline slice access to avoid borrowing the entire `self`
            let start = sample_id * self.pathway_size;
            let sample_slice = &self.current_samples[start..start + self.pathway_size];

            let sample_es_pos = calc_positive_es(&self.ranks, sample_slice);
            let sample_es = calc_es(&self.ranks, sample_slice);

            if sample_es > T::zero() {
                total_pos_es_count += 1;
                self.is_pos_es_buf[sample_id] = true;
            }
            self.stats_buf[sample_id] = (sample_es_pos, sample_id);
        }

        self.stats_buf[..self.sample_size].sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let half_size = self.sample_size / 2;
        self.enrichment_scores.reserve(half_size);
        self.prob_corrector.reserve(half_size);

        let mut sample_id = 0;
        while 2 * sample_id < self.sample_size {
            let (es_val, orig_idx) = self.stats_buf[sample_id];
            self.enrichment_scores.push(es_val);
            if self.is_pos_es_buf[orig_idx] {
                total_pos_es_count -= 1;
            }
            self.prob_corrector.push(total_pos_es_count as usize);
            sample_id += 1;
        }

        // clear the next_samples buffer (sets length to 0, keeps capacity)
        self.next_samples.clear();

        let mut sample_id = 0;
        while 2 * sample_id < self.sample_size - 2 {
            let target_idx = self.stats_buf[self.sample_size - 1 - sample_id].1;

            // explicitly split borrows: read from current_samples, write to next_samples
            let start = target_idx * self.pathway_size;
            let target_slice = &self.current_samples[start..start + self.pathway_size];

            for _ in 0..2 {
                self.next_samples.extend_from_slice(target_slice);
            }
            sample_id += 1;
        }

        let median_idx = self.stats_buf[self.sample_size >> 1].1;
        let start = median_idx * self.pathway_size;
        let median_slice = &self.current_samples[start..start + self.pathway_size];

        self.next_samples.extend_from_slice(median_slice);

        // swap the pointers: Instant, zero-copy, zero-allocation update
        std::mem::swap(&mut self.current_samples, &mut self.next_samples);

        self.sample_size = self.current_samples.len() / self.pathway_size;
    }

    /// Attempts to improve a sample by swapping genes in/out using perturbation
    ///
    /// ### Params
    ///
    /// * `ranks` - Gene ranks
    /// * `k` - Number of genes in sample
    /// * `sample_chunks` - Sample chunks to modify
    /// * `bound` - ES boundary threshold
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Number of successful perturbations
    #[allow(unused_assignments)]
    fn perturbate(
        &self,
        ranks: &[T],
        k: i32,
        sample_chunks: &mut SampleChunks<T>,
        bound: T,
        rng: &mut StdRng,
    ) -> i32 {
        let pert_prmtr = T::from_f64(0.1).unwrap();
        let n = ranks.len() as i32;
        let uid_n = Uniform::new(0, n).unwrap();
        let uid_k = Uniform::new(0, k).unwrap();

        let mut ns: T = sample_chunks
            .chunk_sum
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x);

        let q1 = T::one() / T::from_i32(n - k).unwrap();
        let iters = std::cmp::max(1, (T::from_i32(k).unwrap() * pert_prmtr).to_i32().unwrap());
        let mut moves = 0;

        let mut cand_val = -1;
        let mut has_cand = false;
        let mut cand_x = 0;
        let mut cand_y = T::zero();

        for _ in 0..iters {
            let old_ind = uid_k.sample(rng);
            let mut old_chunk_ind = 0;
            let mut old_ind_in_chunk = 0;
            let old_val;

            {
                let mut tmp = old_ind;
                while old_chunk_ind < sample_chunks.chunks.len()
                    && sample_chunks.chunks[old_chunk_ind].len() <= tmp as usize
                {
                    tmp -= sample_chunks.chunks[old_chunk_ind].len() as i32;
                    old_chunk_ind += 1;
                }
                old_ind_in_chunk = tmp;
                old_val = sample_chunks.chunks[old_chunk_ind][old_ind_in_chunk as usize];
            }

            let new_val = uid_n.sample(rng);

            let new_chunk_ind = match self.chunk_last_element.binary_search(&(new_val)) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            };

            let new_ind_in_chunk =
                match sample_chunks.chunks[new_chunk_ind].binary_search(&(new_val)) {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };

            if new_ind_in_chunk < sample_chunks.chunks[new_chunk_ind].len()
                && sample_chunks.chunks[new_chunk_ind][new_ind_in_chunk] == new_val
            {
                if new_val == old_val {
                    moves += 1;
                }
                continue;
            }

            sample_chunks.chunks[old_chunk_ind].remove(old_ind_in_chunk as usize);
            let adjust =
                if old_chunk_ind == new_chunk_ind && old_ind_in_chunk < new_ind_in_chunk as i32 {
                    1
                } else {
                    0
                };
            sample_chunks.chunks[new_chunk_ind].insert(new_ind_in_chunk - adjust, new_val);

            ns = ns - ranks[old_val as usize] + ranks[new_val as usize];
            sample_chunks.chunk_sum[old_chunk_ind] -= ranks[old_val as usize];
            sample_chunks.chunk_sum[new_chunk_ind] += ranks[new_val as usize];

            if has_cand {
                match old_val.cmp(&cand_val) {
                    std::cmp::Ordering::Equal => {
                        has_cand = false;
                    }
                    std::cmp::Ordering::Less => {
                        cand_x += 1;
                        cand_y -= ranks[old_val as usize];
                    }
                    std::cmp::Ordering::Greater => {}
                }

                if new_val < cand_val {
                    cand_x -= 1;
                    cand_y += ranks[new_val as usize];
                }
            }

            let q2 = T::one() / ns;

            if has_cand && -q1 * T::from_i32(cand_x).unwrap() + q2 * cand_y > bound {
                moves += 1;
                continue;
            }

            let mut cur_x = 0;
            let mut cur_y = T::zero();
            let mut ok = false;
            let mut last = -1;

            for i in 0..sample_chunks.chunks.len() {
                if q2 * (cur_y + sample_chunks.chunk_sum[i]) - q1 * T::from_i32(cur_x).unwrap()
                    < bound
                {
                    cur_y += sample_chunks.chunk_sum[i];
                    cur_x += self.chunk_last_element[i]
                        - last
                        - 1
                        - sample_chunks.chunks[i].len() as i32;
                    last = self.chunk_last_element[i] - 1;
                } else {
                    for &pos in &sample_chunks.chunks[i] {
                        cur_y += ranks[pos as usize];
                        cur_x += pos - last - 1;
                        if q2 * cur_y - q1 * T::from_i32(cur_x).unwrap() > bound {
                            ok = true;
                            has_cand = true;
                            cand_x = cur_x;
                            cand_y = cur_y;
                            cand_val = pos;
                            break;
                        }
                        last = pos;
                    }
                    if ok {
                        break;
                    }
                    cur_x += self.chunk_last_element[i] - 1 - last;
                    last = self.chunk_last_element[i] - 1;
                }
            }

            if !ok {
                ns = ns - ranks[new_val as usize] + ranks[old_val as usize];
                sample_chunks.chunk_sum[old_chunk_ind] += ranks[old_val as usize];
                sample_chunks.chunk_sum[new_chunk_ind] -= ranks[new_val as usize];

                sample_chunks.chunks[new_chunk_ind].remove(new_ind_in_chunk - adjust);
                sample_chunks.chunks[old_chunk_ind].insert(old_ind_in_chunk as usize, old_val);

                if has_cand {
                    if new_val == cand_val {
                        has_cand = false;
                    } else if old_val < cand_val {
                        cand_x -= 1;
                        cand_y += ranks[old_val as usize];
                    }

                    if new_val < cand_val {
                        cand_x += 1;
                        cand_y -= ranks[new_val as usize];
                    }
                }
            } else {
                moves += 1;
            }
        }

        moves
    }

    /// Extends the ES distribution to include the target ES value
    ///
    /// Uses an adaptive sampling approach to explore higher ES values.
    /// Memory structure has been refactored for cache-locality.
    ///
    /// ### Params
    ///
    /// * `es` - Target enrichment score
    /// * `seed` - Random seed
    /// * `eps` - Precision parameter (0.0 for no precision requirement)
    fn extend(&mut self, es: T, seed: u64, eps: T) {
        let mut rng = StdRng::seed_from_u64(seed);

        // Bootstrap the initial random samples
        for sample_id in 0..self.sample_size {
            let mut sample = combination(0, self.ranks.len() - 1, self.pathway_size, &mut rng);
            sample.sort_unstable();
            let _ = calc_es(&self.ranks, &sample);
            self.get_sample_mut(sample_id).copy_from_slice(&sample);
        }

        self.chunks_number = std::cmp::max(
            1,
            T::from_usize(self.pathway_size)
                .unwrap()
                .sqrt()
                .to_i32()
                .unwrap(),
        );
        self.chunk_last_element = vec![0; self.chunks_number as usize];
        self.chunk_last_element[self.chunks_number as usize - 1] = self.ranks.len() as i32;
        let mut tmp: Vec<i32> = vec![0; self.sample_size];
        let mut samples_chunks =
            vec![SampleChunks::new(self.chunks_number as usize); self.sample_size];

        self.duplicate_samples();

        while self.enrichment_scores.last().unwrap_or(&T::zero())
            <= &(es - T::from_f64(1e-10).unwrap())
        {
            for i in 0..self.chunks_number - 1 {
                let pos = (self.pathway_size as i32 + i) / self.chunks_number;

                // Map from the flattened array
                for j in 0..self.sample_size {
                    tmp[j] = self.get_sample(j)[pos as usize] as i32;
                }

                nth_element(&mut tmp, self.sample_size / 2);
                self.chunk_last_element[i as usize] = tmp[self.sample_size / 2];
            }

            for i in 0..self.sample_size {
                for j in 0..self.chunks_number as usize {
                    samples_chunks[i].chunk_sum[j] = T::zero();
                    samples_chunks[i].chunks[j].clear();
                }

                let mut cnt = 0;
                let current_slice = self.get_sample(i);
                for &pos in current_slice {
                    while cnt < self.chunk_last_element.len()
                        && self.chunk_last_element[cnt] <= pos as i32
                    {
                        cnt += 1;
                    }
                    samples_chunks[i].chunks[cnt].push(pos as i32);
                    samples_chunks[i].chunk_sum[cnt] += self.ranks[pos];
                }
            }

            let mut moves = 0;
            while moves < (self.sample_size * self.pathway_size) as i32 {
                for sample_id in 0..self.sample_size {
                    moves += self.perturbate(
                        &self.ranks,
                        self.pathway_size as i32,
                        &mut samples_chunks[sample_id],
                        *self.enrichment_scores.last().unwrap_or(&T::zero()),
                        &mut rng,
                    );
                }
            }

            let chunks_num = self.chunks_number as usize;

            // Write back to the flattened memory array
            for i in 0..self.sample_size {
                let mut offset = 0;
                let dest_slice = self.get_sample_mut(i);
                for j in 0..chunks_num {
                    for &val in &samples_chunks[i].chunks[j] {
                        dest_slice[offset] = val as usize;
                        offset += 1;
                    }
                }
            }

            let prev_top_score = *self.enrichment_scores.last().unwrap_or(&T::zero());
            self.duplicate_samples();

            if self.enrichment_scores.last().unwrap_or(&T::zero()) <= &prev_top_score {
                break;
            }

            if eps != T::zero() {
                let k = self.enrichment_scores.len() / (self.sample_size.div_ceil(2));
                if T::from_usize(k).unwrap() > -T::from_f64(0.5).unwrap() * eps.log2() {
                    break;
                }
            }
        }
    }

    /// Calculate the p-value for a given enrichment score
    ///
    /// ### Params
    ///
    /// * `es` - Enrichment score
    /// * `sign` - Whether to consider sign in calculation
    ///
    /// # Returns
    ///
    /// Tuple of (p-value, error quality flag)
    fn get_pval(&self, es: T, sign: bool) -> (T, bool) {
        #[allow(clippy::manual_div_ceil)]
        let half_size = (self.original_sample_size + 1) / 2;
        let it_index;
        let mut good_error = true;

        if es >= *self.enrichment_scores.last().unwrap_or(&T::zero()) {
            it_index = self.enrichment_scores.len() - 1;
            if es > self.enrichment_scores[it_index] + T::from_f64(1e-10).unwrap() {
                good_error = false;
            }
        } else {
            it_index = match self.enrichment_scores.binary_search_by(|probe| {
                probe.partial_cmp(&es).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(index) => index,
                Err(index) => index,
            };
        }

        let indx = if it_index > 0 { it_index } else { 0 };
        let k = indx / half_size;
        let remainder = self.original_sample_size - (indx % half_size);

        let adj_log = beta_mean_log(half_size, self.original_sample_size);
        let adj_log_pval = T::from_f64(
            k as f64 * adj_log + beta_mean_log(remainder + 1, self.original_sample_size),
        )
        .unwrap();

        if sign {
            (T::zero().max(T::one().min(adj_log_pval.exp())), good_error)
        } else {
            let correction =
                calc_log_correction(&self.prob_corrector, indx, self.original_sample_size);
            let res_log = adj_log_pval + T::from_f64(correction.0).unwrap();
            (
                T::zero().max(T::one().min(res_log.exp())),
                good_error && correction.1,
            )
        }
    }
}

////////////////////
// Main functions //
////////////////////

/// Transform batch results into final GSEA results (es, nes, pval and size)
///
/// ### Params
///
/// * `pathway_scores` - Enrichment scores for pathways
/// * `pathway_sizes` - Sizes of pathways
/// * `gsea_res` - Batch results from permutations
///
/// ### Returns
///
/// Final GSEA results structure
pub fn calculate_nes_es_pval<'a, T: BixverseFloat>(
    pathway_scores: &'a [T],
    pathway_sizes: &'a [usize],
    gsea_res: &GseaBatchResults<T>,
) -> GseaResults<'a, T> {
    let le_zero_mean: Vec<T> = gsea_res
        .le_zero_sum
        .iter()
        .zip(gsea_res.le_zero.iter())
        .map(|(a, b)| *a / T::from_usize(*b).unwrap())
        .collect();

    let ge_zero_mean: Vec<T> = gsea_res
        .ge_zero_sum
        .iter()
        .zip(gsea_res.ge_zero.iter())
        .map(|(a, b)| *a / T::from_usize(*b).unwrap())
        .collect();

    let nes: Vec<Option<T>> = pathway_scores
        .iter()
        .zip(ge_zero_mean.iter().zip(le_zero_mean.iter()))
        .map(|(&score, (&ge_mean, &le_mean))| {
            if (score > T::zero() && ge_mean != T::zero())
                || (score < T::zero() && le_mean != T::zero())
            {
                Some(
                    score
                        / if score > T::zero() {
                            ge_mean
                        } else {
                            le_mean.abs()
                        },
                )
            } else {
                None
            }
        })
        .collect();

    let pvals: Vec<T> = gsea_res
        .le_es
        .iter()
        .zip(gsea_res.le_zero.iter())
        .zip(gsea_res.ge_es.iter())
        .zip(gsea_res.ge_zero.iter())
        .map(|(((le_es, le_zero), ge_es), ge_zero)| {
            (T::from_usize(1 + le_es).unwrap() / T::from_usize(1 + le_zero).unwrap())
                .min(T::from_usize(1 + ge_es).unwrap() / T::from_usize(1 + ge_zero).unwrap())
        })
        .collect();

    let n_more_extreme: Vec<usize> = pathway_scores
        .iter()
        .zip(gsea_res.ge_es.iter())
        .zip(gsea_res.le_es.iter())
        .map(|((es, ge_es), le_es)| if es > &T::zero() { *ge_es } else { *le_es })
        .collect();

    GseaResults {
        es: pathway_scores,
        nes,
        pvals,
        n_more_extreme,
        le_zero: gsea_res.le_zero.clone(),
        ge_zero: gsea_res.ge_zero.clone(),
        size: pathway_sizes,
    }
}

////////////////////////////////////////////
// Classical Gene Set enrichment analysis //
////////////////////////////////////////////

/// Calculate the Enrichment score assuming stats is sorted and pathway contains index positions
///
/// ### Params
///
/// * `stats` - Sorted gene statistics
/// * `pathway` - Index positions of genes in the pathway
///
/// ### Returns
///
/// Enrichment score
pub fn calculate_es<T: BixverseFloat>(stats: &[T], pathway: &[usize]) -> T {
    let no_genes = stats.len();
    let p_total = pathway.len();
    let mut nr = T::zero();
    for p in pathway {
        nr += stats[*p].abs()
    }
    let weight_hit = T::one() / nr;
    let weight_miss = T::one() / T::from_usize(no_genes - p_total).unwrap();
    let mut running_sum = T::zero();
    let mut max_run_sum = T::zero();
    let mut min_run_sum = T::zero();
    for (i, x) in stats.iter().enumerate() {
        if pathway.contains(&i) {
            running_sum += x.abs() * weight_hit
        } else {
            running_sum -= weight_miss
        }
        max_run_sum = running_sum.max(max_run_sum);
        min_run_sum = running_sum.min(min_run_sum);
    }
    if max_run_sum > min_run_sum.abs() {
        max_run_sum
    } else {
        min_run_sum
    }
}

/// Calculate once for each size the permutation-based enrichment scores
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `gene_set_sizes` - Unique gene set sizes
/// * `shared_perms` - Shared permutations
///
/// ### Returns
///
/// HashMap mapping sizes to permutation scores
fn create_perm_es<T: BixverseFloat>(
    stats: &[T],
    gene_set_sizes: &[usize],
    shared_perms: &[Vec<usize>],
) -> FxHashMap<usize, Vec<T>> {
    let mut shared_perm_es =
        FxHashMap::with_capacity_and_hasher(gene_set_sizes.len(), FxBuildHasher);
    for size in gene_set_sizes {
        let perm_es: Vec<T> = shared_perms
            .into_par_iter()
            .map(|perm| calculate_es(stats, &perm[..*size]))
            .collect();
        shared_perm_es.insert(*size, perm_es);
    }
    shared_perm_es
}

/// Calculate the permutations in the 'traditional' way
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `pathway_scores` - Pathway enrichment scores
/// * `pathway_sizes` - Pathway sizes
/// * `iters` - Number of iterations
/// * `seed` - Random seed
///
/// ### Returns
///
/// Batch results from traditional, permutation-based method
pub fn calc_gsea_stat_traditional_batch<T: BixverseFloat>(
    stats: &[T],
    pathway_scores: &[T],
    pathway_sizes: &[usize],
    iters: usize,
    seed: u64,
) -> GseaBatchResults<T> {
    let n = stats.len();
    let k = array_max(pathway_sizes);
    let k_unique = unique(pathway_sizes);

    let m = pathway_scores.len();

    let shared_perm = create_random_gs_indices(iters, k, n, seed, false);

    let shared_perm_es = create_perm_es(stats, &k_unique, &shared_perm);

    let mut le_es = Vec::with_capacity(m);
    let mut ge_es = Vec::with_capacity(m);
    let mut le_zero = Vec::with_capacity(m);
    let mut ge_zero = Vec::with_capacity(m);
    let mut le_zero_sum = Vec::with_capacity(m);
    let mut ge_zero_sum = Vec::with_capacity(m);

    for (es, size) in pathway_scores.iter().zip(pathway_sizes.iter()) {
        let random_es = shared_perm_es.get(size).unwrap();
        let le_es_i = random_es.iter().filter(|x| x <= &es).count();
        let ge_es_i = iters - le_es_i;
        let le_zero_i = random_es.iter().filter(|x| x <= &&T::zero()).count();
        let ge_zero_i = iters - le_zero_i;
        let le_zero_sum_i: T = random_es
            .iter()
            .map(|&x| x.min(T::zero()))
            .fold(T::zero(), |acc, x| acc + x);
        let ge_zero_sum_i: T = random_es
            .iter()
            .map(|&x| x.max(T::zero()))
            .fold(T::zero(), |acc, x| acc + x);
        le_es.push(le_es_i);
        ge_es.push(ge_es_i);
        le_zero.push(le_zero_i);
        ge_zero.push(ge_zero_i);
        le_zero_sum.push(le_zero_sum_i);
        ge_zero_sum.push(ge_zero_sum_i);
    }

    GseaBatchResults {
        le_es,
        ge_es,
        le_zero,
        ge_zero,
        le_zero_sum,
        ge_zero_sum,
    }
}

//////////////////
// FGSEA simple //
//////////////////

/// Square root heuristic approximation from fgsea paper with convex hull updates
///
/// Selected stats needs to be one-indexed!!!
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `selected_stats` - Selected gene indices (one-indexed)
/// * `selected_order` - Order of selected genes
/// * `gsea_param` - GSEA parameter for weighting
/// * `rev` - Whether to reverse direction
///
/// # Returns
///
/// Vector of enrichment scores
fn gsea_stats_sq<T>(
    stats: &[T],
    selected_stats: &[usize],
    selected_order: &[usize],
    gsea_param: T,
    rev: bool,
) -> Vec<T>
where
    T: BixverseFloat + Default,
{
    let n = stats.len() as i32;
    let k = selected_stats.len();
    let mut nr = T::zero();

    let mut res = vec![T::zero(); k];

    let mut xs = SegmentTree::<i32>::new(k + 1);
    let mut ys = SegmentTree::<T>::new(k + 1);

    let mut selected_ranks = ranks_from_order(selected_order);

    if !rev {
        let mut prev: i32 = -1;
        for (i, _) in (0..k).enumerate() {
            let j = selected_order[i];
            let t = (selected_stats[j] - 1) as i32;
            assert!(t - prev >= 1);
            xs.increment(i, t - prev);
            prev = t;
        }
        xs.increment(k, n - 1 - prev);
    } else {
        let mut prev = n;
        for i in 0..k {
            selected_ranks[i] = k as i32 - 1 - selected_ranks[i];
            let j = selected_order[k - 1 - i];
            let t = (selected_stats[j] - 1) as i32;
            assert!(prev - t >= 1);
            xs.increment(i, prev - t);
            prev = t;
        }
        xs.increment(k, prev);
    }

    let mut st_prev = vec![0; k + 2];
    let mut st_next = vec![0; k + 2];

    let k1 = std::cmp::max(1, T::from_usize(k + 1).unwrap().sqrt().to_usize().unwrap());
    let k2 = (k + 1) / k1 + 1;

    let mut block_summit = vec![0; k2];
    let mut block_start = vec![0; k2];
    let mut block_end = vec![0; k2];

    for i in (0..=k + 1).step_by(k1) {
        let block = i / k1;
        block_start[block] = i;
        block_end[block] = std::cmp::min(i + k1 - 1, k + 1);

        for j in 1..=block_end[block] - i {
            st_prev[i + j] = i + j - 1;
            st_next[i + j - 1] = i + j;
        }
        st_prev[i] = i;
        st_next[block_end[block]] = block_end[block];
        block_summit[block] = block_end[block];
    }

    let mut stat_eps = T::from_f64(1e-5).unwrap();
    for (i, _) in (0..k).enumerate() {
        let t = selected_stats[i];
        let xx = stats[t].abs();
        if xx > T::zero() {
            stat_eps = stat_eps.min(xx);
        }
    }
    stat_eps /= T::from_f64(1024.0).unwrap();

    for i in 0..k {
        let t = selected_stats[i] - 1;
        let t_rank = selected_ranks[i] as usize;

        let adj_stat = stats[t].abs().max(stat_eps).powf(gsea_param);

        xs.increment(t_rank, -1);
        ys.increment(t_rank, adj_stat);
        nr += adj_stat;

        let m = i + 1;

        let cur_block = (t_rank + 1) / k1;
        let bs = block_start[cur_block];
        let be = block_end[cur_block];

        let mut cur_top = t_rank.max(bs);

        for j in t_rank + 1..=be {
            let c = j;

            let xc = T::from_i32(xs.query_r(c)).unwrap();
            let yc = ys.query_r(c);

            let mut b = cur_top;

            let mut xb = T::from_i32(xs.query_r(b)).unwrap();
            let mut yb = ys.query_r(b);

            while st_prev[cur_top] != cur_top {
                let a = st_prev[cur_top];

                let xa = T::from_i32(xs.query_r(a)).unwrap();
                let ya = ys.query_r(a);

                let pr = (xb - xa) * (yc - yb) - (yb - ya) * (xc - xb);

                let pr = if yc - ya < T::from_f64(1e-13).unwrap() {
                    T::zero()
                } else {
                    pr
                };

                if pr <= T::zero() {
                    break;
                }
                cur_top = a;
                st_next[b] = usize::MAX;
                b = a;
                xb = xa;
                yb = ya;
            }

            st_prev[c] = cur_top;
            st_next[cur_top] = c;
            cur_top = c;

            if st_next[c] != usize::MAX {
                break;
            }
        }

        let coef = T::from_i32(n - m as i32).unwrap() / nr;

        let mut max_p = T::zero();

        block_summit[cur_block] = cur_top.max(block_summit[cur_block]);

        for (block, _) in (0..k2).collect::<std::vec::Vec<usize>>().iter().enumerate() {
            let mut cur_summit = block_summit[block];

            let mut cur_dist =
                ys.query_r(cur_summit) * coef - T::from_i32(xs.query_r(cur_summit)).unwrap();

            loop {
                let next_summit = st_prev[cur_summit];
                let next_dist =
                    ys.query_r(next_summit) * coef - T::from_i32(xs.query_r(next_summit)).unwrap();

                if next_dist <= cur_dist {
                    break;
                }
                cur_dist = next_dist;
                cur_summit = next_summit;
            }

            block_summit[block] = cur_summit;
            max_p = max_p.max(if cur_dist.is_sign_negative() {
                T::zero()
            } else {
                cur_dist
            });
        }

        max_p /= T::from_i32(n - m as i32).unwrap();

        res[i] = max_p;
    }

    res
}

/// Calculate cumulative enrichment scores via block-wise approximation
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `selected_stats` - Selected gene indices
/// * `gsea_param` - GSEA parameter for weighting
///
/// ### Returns
///
/// Vector of cumulative enrichment scores
pub fn calc_gsea_stat_cumulative<T>(stats: &[T], selected_stats: &[usize], gsea_param: T) -> Vec<T>
where
    T: BixverseFloat + Default,
{
    let selected_order = fgsea_order(selected_stats);

    let res = gsea_stats_sq(stats, selected_stats, &selected_order, gsea_param, false);

    let res_down = gsea_stats_sq(stats, selected_stats, &selected_order, gsea_param, true);

    res.into_iter()
        .zip(res_down)
        .map(|(up, down)| {
            if up == down {
                T::zero()
            } else if up < down {
                -down
            } else {
                up
            }
        })
        .collect()
}

/// Create permutations for the fgsea simple method
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `gsea_param` - GSEA parameter
/// * `iters` - Number of iterations
/// * `max_len` - Maximum pathway length
/// * `universe_length` - Total number of genes
/// * `seed` - Random seed
/// * `one_indexed` - Whether to use 1-based indexing
///
/// ### Returns
///
/// Vector of permutation enrichment score vectors
pub fn create_perm_es_simple<T>(
    stats: &[T],
    gsea_param: T,
    iters: usize,
    max_len: usize,
    universe_length: usize,
    seed: u64,
    one_indexed: bool,
) -> Vec<Vec<T>>
where
    T: BixverseFloat + Default,
{
    let shared_perm = create_random_gs_indices(iters, max_len, universe_length, seed, one_indexed);

    let chunk_size = std::cmp::max(1, iters / (rayon::current_num_threads() * 4));

    let rand_es: Vec<Vec<T>> = shared_perm
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_results = Vec::with_capacity(chunk.len());

            for i in chunk {
                let x = calc_gsea_stat_cumulative(stats, &i, gsea_param);
                local_results.push(x)
            }

            local_results
        })
        .collect();

    rand_es
}

/// Abstraction wrapper to be used in different parts of the package
///
/// ### Params
///
/// * `pathway_scores` - Pathway enrichment scores
/// * `pathway_sizes` - Pathway sizes
/// * `shared_perm` - Shared permutation results
///
/// ### Returns
///
/// Batch results from permutation analysis
pub fn calc_gsea_stats_wrapper<T: BixverseFloat>(
    pathway_scores: &[T],
    pathway_sizes: &[usize],
    shared_perm: &Vec<Vec<T>>,
) -> GseaBatchResults<T> {
    let m = pathway_scores.len();

    let mut le_es = vec![0; m];
    let mut ge_es = vec![0; m];
    let mut le_zero = vec![0; m];
    let mut ge_zero = vec![0; m];
    let mut le_zero_sum = vec![T::zero(); m];
    let mut ge_zero_sum = vec![T::zero(); m];

    for rand_es_i in shared_perm {
        let rand_es_p = subvector(rand_es_i, pathway_sizes).unwrap();

        for i in 0..m {
            if rand_es_p[i] <= pathway_scores[i] {
                le_es[i] += 1;
            } else {
                ge_es[i] += 1;
            }

            if rand_es_p[i] <= T::zero() {
                le_zero[i] += 1;
                le_zero_sum[i] += rand_es_p[i];
            } else {
                ge_zero[i] += 1;
                ge_zero_sum[i] += rand_es_p[i];
            }
        }
    }

    GseaBatchResults {
        le_es,
        ge_es,
        le_zero,
        ge_zero,
        le_zero_sum,
        ge_zero_sum,
    }
}

/// Calculate random scores batch-wise using the fgsea simple method
///
/// ### Params
///
/// * `stats` - Gene statistics
/// * `pathway_scores` - Pathway enrichment scores
/// * `pathway_sizes` - Pathway sizes
/// * `iters` - Number of iterations
/// * `gsea_param` - GSEA parameter
/// * `seed` - Random seed
///
/// ### Returns
///
/// Batch results from fgsea simple method
pub fn calc_gsea_stat_cumulative_batch<T>(
    stats: &[T],
    pathway_scores: &[T],
    pathway_sizes: &[usize],
    iters: usize,
    gsea_param: T,
    seed: u64,
) -> GseaBatchResults<T>
where
    T: BixverseFloat + Default,
{
    let n = stats.len();
    let k = array_max(pathway_sizes);

    let shared_perm = create_perm_es_simple(stats, gsea_param, iters, k, n, seed, true);

    calc_gsea_stats_wrapper(pathway_scores, pathway_sizes, &shared_perm)
}

//////////////////////
// FGSEA multilevel //
//////////////////////

/// Calculates multilevel error for a given p-value and sample size
///
/// ### Params
///
/// * `pval` - P-value
/// * `sample_size` - Sample size
///
/// ### Returns
///
/// Multilevel error estimate
fn multilevel_error<T: BixverseFloat>(pval: &T, sample_size: &T) -> T {
    let floor_term = (-pval.log2() + T::one()).floor();
    let trigamma_diff = trigamma((*sample_size + T::one()) / T::from_f64(2.0).unwrap())
        - trigamma(*sample_size + T::one());

    (floor_term * trigamma_diff).sqrt() / T::from_f64(f64::ln(2.0)).unwrap()
}

/// Function to do the multi-level magic in fgsea
///
/// ### Params
///
/// * `enrichment_score` - Target enrichment score
/// * `ranks` - Gene ranks
/// * `pathway_size` - Size of pathway
/// * `sample_size` - Sample size
/// * `seed` - Random seed
/// * `eps` - Precision parameter (0.0 for no precision requirement)
/// * `sign` - Whether to consider sign in calculation
///
/// ### Returns
///
/// Tuple of (p-value, error quality flag)
#[allow(clippy::too_many_arguments)]
pub fn fgsea_multilevel_helper<T: BixverseFloat>(
    enrichment_score: T,
    ranks: &[T],
    pathway_size: usize,
    sample_size: usize,
    seed: u64,
    eps: T,
    sign: bool,
) -> (T, bool) {
    let ranks: Vec<T> = if enrichment_score >= T::zero() {
        ranks.iter().map(|&r| r.abs()).collect()
    } else {
        let mut neg_ranks: Vec<T> = ranks.iter().map(|&r| r.abs()).collect();
        neg_ranks.reverse();
        neg_ranks
    };

    let mut es_ruler = EsRuler::new(&ranks, sample_size, pathway_size);
    es_ruler.extend(enrichment_score.abs(), seed, eps);
    es_ruler.get_pval(enrichment_score.abs(), sign)
}

/// Calculates the simple and multi error estimates
///
/// ### Params
///
/// * `n_more_extreme` - Number of more extreme permutations
/// * `nperm` - Total permutations
/// * `sample_size` - Sample size
///
/// ### Returns
///
/// Tuple of (simple errors, multi errors)
pub fn calc_simple_and_multi_error<T: BixverseFloat>(
    n_more_extreme: &[usize],
    nperm: usize,
    sample_size: usize,
) -> MultiLevelErrRes<T> {
    let no_tests = n_more_extreme.len();
    let n_more_extreme_f: Vec<T> = n_more_extreme
        .iter()
        .map(|x| T::from_usize(*x).unwrap())
        .collect();
    let nperm_f = T::from_usize(nperm).unwrap();
    let sample_size_f = T::from_usize(sample_size).unwrap();

    let mut left_border = Vec::with_capacity(no_tests);
    let mut right_border = Vec::with_capacity(no_tests);
    let mut crude_est = Vec::with_capacity(no_tests);
    for n in &n_more_extreme_f {
        if n > &T::zero() {
            let beta = Beta::new(
                n.to_f64().unwrap(),
                (nperm_f - *n + T::one()).to_f64().unwrap(),
            )
            .unwrap();
            left_border.push(T::from_f64(beta.inverse_cdf(0.025).log2()).unwrap());
        } else {
            left_border.push(T::neg_infinity());
        }
        let beta = Beta::new(
            (*n + T::one()).to_f64().unwrap(),
            (nperm_f - *n).to_f64().unwrap(),
        )
        .unwrap();
        right_border.push(T::from_f64(beta.inverse_cdf(1.0 - 0.025).log2()).unwrap());
        crude_est.push(((*n + T::one()) / (nperm_f + T::one())).log2());
    }

    let mut simple_err = Vec::with_capacity(no_tests);
    for i in 0..no_tests {
        simple_err.push(
            T::from_f64(0.5).unwrap()
                * (crude_est[i] - left_border[i]).max(right_border[i] - crude_est[i]),
        );
    }

    let multi_err: Vec<T> = n_more_extreme_f
        .iter()
        .map(|n| {
            let pval = (*n + T::one()) / (nperm_f + T::one());
            multilevel_error(&pval, &sample_size_f)
        })
        .collect();

    (simple_err, multi_err)
}
