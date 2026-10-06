//! CellPhoneDB statistical analysis of ligand/receptor interactions between
//! cell clusters.
//!
//! Mirrors the CellPhoneDB v5 numerics (`cpdb_statistical_analysis_helper.py`)
//! exactly. The caller supplies the L/R database already resolved to gene
//! indices. A complex takes the minimum over its subunits for both the mean
//! and the fraction expressing. The null shuffles cluster labels globally
//! across the included cells, and `p = #(perm > real) / n_perm`, with no
//! pseudo-count, as in the reference.
//!
//! The L/R genes are read once from the gene-major store. The permutations
//! then run in memory over the non-zeros only, in batches that share one
//! pass over each gene's data.

use faer::Mat;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::Separable;

use crate::prelude::*;
use crate::single_cell::sc_utils::utils_tree::tree_seed;

///////////////
// Constants //
///////////////

/// Permutations sharing one pass over the gene data. Labels for a batch are
/// stored cell-major (`[cell][perm]`), so one cache line serves every
/// permutation in the batch. At 16 and u16 labels, that is 32 bytes per cell.
/// Not tuned yet; change it via [`CellPhoneDbParams::perm_batch`].
const DEFAULT_PERM_BATCH: usize = 16;

////////////
// Params //
////////////

/// A ligand/receptor interaction resolved to gene indices.
///
/// The orientation is kept as given: in the cluster pair `(A, B)`,
/// `partner_a` is measured in `A` and `partner_b` in `B`.
#[derive(Clone, Debug)]
pub struct LrInteraction {
    /// Subunit gene indices of partner a. A single gene is a length one list.
    pub partner_a: Vec<usize>,
    /// Subunit gene indices of partner b. A single gene is a length one list.
    pub partner_b: Vec<usize>,
}

/// Parameters for the CellPhoneDB statistical analysis.
#[derive(Clone, Debug)]
pub struct CellPhoneDbParams<T> {
    /// Number of label permutations.
    pub n_perm: usize,
    /// Both partners need a fraction expressing strictly above this value.
    pub threshold: T,
    /// Seed for the permutations.
    pub seed: usize,
    /// Permutations per batch. `None` uses [`DEFAULT_PERM_BATCH`].
    pub perm_batch: Option<usize>,
}

impl<T: BixverseFloat> Default for CellPhoneDbParams<T> {
    /// Defaults match CellPhoneDB: 1000 permutations, threshold 0.1.
    fn default() -> Self {
        Self {
            n_perm: 1000,
            threshold: T::from_f64(0.1).unwrap(),
            seed: 42,
            perm_batch: None,
        }
    }
}

impl<T: BixverseFloat> CellPhoneDbParams<T> {
    /// Generate new parameters.
    ///
    /// ### Params
    ///
    /// * `n_perm` - Number of label permutations.
    /// * `threshold` - Minimum fraction of cells expressing (strict).
    /// * `seed` - Seed for the permutations.
    /// * `perm_batch` - Permutations per batch. `None` uses the default.
    ///
    /// ### Returns
    ///
    /// Self.
    pub fn new(n_perm: usize, threshold: T, seed: usize, perm_batch: Option<usize>) -> Self {
        Self {
            n_perm,
            threshold,
            seed,
            perm_batch,
        }
    }
}

/// Resolve the permutation batch size.
///
/// ### Params
///
/// * `perm_batch` - User override.
/// * `n_perm` - Total number of permutations.
///
/// ### Returns
///
/// The batch size, at least one and at most `n_perm`.
fn resolve_perm_batch(perm_batch: Option<usize>, n_perm: usize) -> usize {
    perm_batch
        .unwrap_or(DEFAULT_PERM_BATCH)
        .clamp(1, n_perm.max(1))
}

/////////////
// Results //
/////////////

/// Observed statistics, without permutations.
///
/// This is the CellPhoneDB "simple" analysis. It also feeds the DEG gate.
#[derive(Clone, Debug)]
pub struct CellPhoneDbObs<T> {
    /// Unique, ascending gene indices over all partners. Rows of `gene_mean`
    /// and `gene_pct`.
    pub genes: Vec<usize>,
    /// Ordered cluster pairs `(A, B)`. Columns of `means`.
    pub pairs: Vec<(usize, usize)>,
    /// Per-gene, per-cluster mean of the normalised values. Shape
    /// `(genes, clusters)`. This is the deconvoluted table.
    pub gene_mean: Mat<T>,
    /// Per-gene, per-cluster fraction of cells with a value above zero. Shape
    /// `(genes, clusters)`.
    pub gene_pct: Mat<T>,
    /// Interaction means, `(x > 0) * (y > 0) * (x + y) / 2`. Shape
    /// `(interactions, pairs)`.
    pub means: Mat<T>,
    /// Fraction-expressing gate for both partners, row-major
    /// `(interactions, pairs)`.
    pub gate: Vec<bool>,
}

/// Statistical analysis results.
#[derive(Clone, Debug)]
pub struct CellPhoneDbRes<T> {
    /// Observed statistics.
    pub obs: CellPhoneDbObs<T>,
    /// Permutation p-values. Set to 1 where the mean is zero or the gate
    /// fails. Shape `(interactions, pairs)`.
    pub pvals: Mat<T>,
}

/////////////
// Helpers //
/////////////

/// The L/R genes held in memory as CSC over the included cells.
struct LrStore {
    /// Offsets into `cells` and `vals` per gene, length `n_genes + 1`.
    indptr: Vec<usize>,
    /// Position of the cell within the included cells.
    cells: Vec<u32>,
    /// Normalised value, upcast from the stored F16.
    vals: Vec<f32>,
}

/// Everything the observed and permuted passes share.
struct CpdbSetup {
    /// Unique, ascending gene indices.
    genes: Vec<usize>,
    /// The genes over the included cells.
    store: LrStore,
    /// Cluster label per included cell.
    labels: Vec<u16>,
    /// Cells per cluster.
    sizes: Vec<usize>,
    /// Partner a subunits as rows into `genes`, per interaction.
    rows_a: Vec<Vec<usize>>,
    /// Partner b subunits as rows into `genes`, per interaction.
    rows_b: Vec<Vec<usize>>,
    /// Ordered cluster pairs.
    pairs: Vec<(usize, usize)>,
}

/// Validate the inputs, read the L/R genes once and remap cells to their
/// position within the included cells.
///
/// ### Params
///
/// * `reader` - Gene-based reader.
/// * `interactions` - The interactions.
/// * `clusters` - Disjoint cell indices per cluster.
/// * `pairs` - Cluster pairs to test. `None` tests all ordered pairs,
///   `(0, 0), (0, 1), ...`.
///
/// ### Returns
///
/// The [`CpdbSetup`].
fn prepare<S: SingleCellReading>(
    reader: &S,
    interactions: &[LrInteraction],
    clusters: &[Vec<usize>],
    pairs: Option<&[(usize, usize)]>,
) -> Result<CpdbSetup, BixverseErrors> {
    if !reader.is_gene_based() {
        return Err(BixverseErrors::ReaderModeMismatch {
            actual: "cell-based",
            requested: "gene-based",
        });
    }
    let header = reader.get_header();
    let (n_cells, n_genes) = (header.total_cells, header.total_genes);
    let n_clusters = clusters.len();
    if n_clusters > u16::MAX as usize {
        return Err(BixverseErrors::InvalidArgument(format!(
            "CellPhoneDB supports at most {} clusters, got {n_clusters}.",
            u16::MAX
        )));
    }

    let mut genes: Vec<usize> = Vec::new();
    for (i, inter) in interactions.iter().enumerate() {
        if inter.partner_a.is_empty() || inter.partner_b.is_empty() {
            return Err(BixverseErrors::CpdbEmptyPartner { interaction: i });
        }
        for &g in inter.partner_a.iter().chain(inter.partner_b.iter()) {
            if g >= n_genes {
                return Err(BixverseErrors::CpdbGeneIndexOutOfRange { index: g, n_genes });
            }
            genes.push(g);
        }
    }
    genes.sort_unstable();
    genes.dedup();

    let to_rows = |subunits: &[usize]| -> Vec<usize> {
        subunits
            .iter()
            .map(|g| genes.binary_search(g).expect("gene collected above"))
            .collect()
    };
    let rows_a: Vec<Vec<usize>> = interactions.iter().map(|x| to_rows(&x.partner_a)).collect();
    let rows_b: Vec<Vec<usize>> = interactions.iter().map(|x| to_rows(&x.partner_b)).collect();

    let pairs: Vec<(usize, usize)> = match pairs {
        Some(p) => {
            if let Some(&(a, b)) = p.iter().find(|&&(a, b)| a >= n_clusters || b >= n_clusters) {
                return Err(BixverseErrors::InvalidArgument(format!(
                    "CellPhoneDB: cluster pair ({a}, {b}) is outside the {n_clusters} clusters."
                )));
            }
            p.to_vec()
        }
        None => (0..n_clusters)
            .flat_map(|a| (0..n_clusters).map(move |b| (a, b)))
            .collect(),
    };

    let mut position = vec![u32::MAX; n_cells];
    let mut labels: Vec<u16> = Vec::new();
    for (k, cells) in clusters.iter().enumerate() {
        for &c in cells {
            if c >= n_cells {
                return Err(BixverseErrors::CpdbCellIndexOutOfRange { index: c, n_cells });
            }
            if position[c] != u32::MAX {
                return Err(BixverseErrors::CpdbClusterOverlap { cell: c });
            }
            position[c] = labels.len() as u32;
            labels.push(k as u16);
        }
    }
    let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();

    let chunks = reader.read_gene_parallel(&genes)?;
    let mut store = LrStore {
        indptr: Vec::with_capacity(genes.len() + 1),
        cells: Vec::new(),
        vals: Vec::new(),
    };
    store.indptr.push(0);
    for chunk in &chunks {
        for (&c, v) in chunk.indices.iter().zip(chunk.data_norm.iter()) {
            let pos = position[c as usize];
            if pos != u32::MAX {
                store.cells.push(pos);
                store.vals.push(v.to_f32());
            }
        }
        store.indptr.push(store.cells.len());
    }

    Ok(CpdbSetup {
        genes,
        store,
        labels,
        sizes,
        rows_a,
        rows_b,
        pairs,
    })
}

/// Per-gene, per-cluster means for a batch of label vectors.
///
/// Sums in f64 and divides by the fixed cluster sizes. The observed means go
/// through here too, with a batch of one, so observed and permuted values
/// share one summation order and the strict `>` comparison sees ties as
/// ties.
///
/// ### Params
///
/// * `store` - The L/R genes.
/// * `labels` - Cell-major labels, `labels[cell * batch + j]`.
/// * `batch` - Permutations in this batch.
/// * `sizes` - Cells per cluster.
/// * `out` - Output, `out[(g * n_clusters + k) * batch + j]`. Overwritten.
fn group_means_batched(
    store: &LrStore,
    labels: &[u16],
    batch: usize,
    sizes: &[usize],
    out: &mut [f64],
) {
    let n_clusters = sizes.len();
    out.fill(0.0);
    for g in 0..store.indptr.len() - 1 {
        let gene_out = &mut out[g * n_clusters * batch..(g + 1) * n_clusters * batch];
        for idx in store.indptr[g]..store.indptr[g + 1] {
            let cell = store.cells[idx] as usize;
            let v = store.vals[idx] as f64;
            let lab = &labels[cell * batch..(cell + 1) * batch];
            for (j, &k) in lab.iter().enumerate() {
                gene_out[k as usize * batch + j] += v;
            }
        }
    }
    for chunk in out.chunks_exact_mut(n_clusters * batch) {
        for (k, &n) in sizes.iter().enumerate() {
            let inv = if n > 0 { 1.0 / n as f64 } else { 0.0 };
            for v in &mut chunk[k * batch..(k + 1) * batch] {
                *v *= inv;
            }
        }
    }
}

/// Partner value in one cluster: the minimum over subunit gene means.
///
/// ### Params
///
/// * `means` - Output of [`group_means_batched`].
/// * `rows` - Subunit rows.
/// * `k` - Cluster.
/// * `j` - Permutation within the batch.
/// * `n_clusters` - Number of clusters.
/// * `batch` - Permutations in the batch.
///
/// ### Returns
///
/// The partner mean.
#[inline]
fn partner_mean(
    means: &[f64],
    rows: &[usize],
    k: usize,
    j: usize,
    n_clusters: usize,
    batch: usize,
) -> f64 {
    rows.iter()
        .map(|&g| means[(g * n_clusters + k) * batch + j])
        .fold(f64::INFINITY, f64::min)
}

/// The CellPhoneDB interaction mean: zero unless both partners are positive.
///
/// ### Params
///
/// * `x` - Partner a mean in cluster A.
/// * `y` - Partner b mean in cluster B.
///
/// ### Returns
///
/// `(x + y) / 2` if both are positive, else zero.
#[inline]
fn interaction_mean(x: f64, y: f64) -> f64 {
    if x > 0.0 && y > 0.0 {
        (x + y) / 2.0
    } else {
        0.0
    }
}

/// Observed statistics from a prepared setup.
///
/// ### Params
///
/// * `setup` - The prepared inputs.
/// * `threshold` - Minimum fraction expressing (strict).
///
/// ### Returns
///
/// The observed statistics, plus the f64 interaction means (row-major,
/// `(interactions, pairs)`) for the permutation comparison.
fn observed<T: BixverseFloat>(setup: &CpdbSetup, threshold: T) -> (CellPhoneDbObs<T>, Vec<f64>) {
    let n_genes = setup.genes.len();
    let n_clusters = setup.sizes.len();
    let n_inter = setup.rows_a.len();
    let n_pairs = setup.pairs.len();
    let thr = threshold.to_f64().unwrap();

    let mut means = vec![0.0; n_genes * n_clusters];
    group_means_batched(&setup.store, &setup.labels, 1, &setup.sizes, &mut means);

    let mut pct = vec![0.0; n_genes * n_clusters];
    for g in 0..n_genes {
        let st = &setup.store;
        for idx in st.indptr[g]..st.indptr[g + 1] {
            if st.vals[idx] > 0.0 {
                pct[g * n_clusters + setup.labels[st.cells[idx] as usize] as usize] += 1.0;
            }
        }
        for (k, &n) in setup.sizes.iter().enumerate() {
            if n > 0 {
                pct[g * n_clusters + k] /= n as f64;
            }
        }
    }

    let partner_pct = |rows: &[usize], k: usize| -> f64 {
        rows.iter()
            .map(|&g| pct[g * n_clusters + k])
            .fold(f64::INFINITY, f64::min)
    };

    let mut real = vec![0.0; n_inter * n_pairs];
    let mut gate = vec![false; n_inter * n_pairs];
    real.par_chunks_mut(n_pairs)
        .zip(gate.par_chunks_mut(n_pairs))
        .enumerate()
        .for_each(|(i, (real_row, gate_row))| {
            let (ra, rb) = (&setup.rows_a[i], &setup.rows_b[i]);
            for (p, &(a, b)) in setup.pairs.iter().enumerate() {
                let x = partner_mean(&means, ra, a, 0, n_clusters, 1);
                let y = partner_mean(&means, rb, b, 0, n_clusters, 1);
                real_row[p] = interaction_mean(x, y);
                gate_row[p] = partner_pct(ra, a) > thr && partner_pct(rb, b) > thr;
            }
        });

    let to_t = |v: f64| T::from_f64(v).unwrap();
    let obs = CellPhoneDbObs {
        genes: setup.genes.clone(),
        pairs: setup.pairs.clone(),
        gene_mean: Mat::from_fn(n_genes, n_clusters, |g, k| to_t(means[g * n_clusters + k])),
        gene_pct: Mat::from_fn(n_genes, n_clusters, |g, k| to_t(pct[g * n_clusters + k])),
        means: Mat::from_fn(n_inter, n_pairs, |i, p| to_t(real[i * n_pairs + p])),
        gate,
    };
    (obs, real)
}

///////////////
// Analysis //
///////////////

/// CellPhoneDB observed statistics, without permutations.
///
/// This is the CellPhoneDB "simple" analysis, and the input to
/// [`cpdb_deg_gate`].
///
/// ### Params
///
/// * `reader` - Gene-based reader. Uses the normalised layer.
/// * `interactions` - The interactions, resolved to gene indices.
/// * `clusters` - Disjoint cell indices per cluster. Cells in no cluster are
///   ignored.
/// * `pairs` - Cluster pairs to test, e.g. restricted to microenvironments.
///   `None` tests all ordered pairs `(0, 0), (0, 1), ...`.
/// * `threshold` - Minimum fraction expressing (strict).
///
/// ### Returns
///
/// The [`CellPhoneDbObs`].
pub fn cellphonedb_observed<T, S>(
    reader: &S,
    interactions: &[LrInteraction],
    clusters: &[Vec<usize>],
    pairs: Option<&[(usize, usize)]>,
    threshold: T,
) -> Result<CellPhoneDbObs<T>, BixverseErrors>
where
    T: BixverseFloat + Send + Sync,
    S: SingleCellReading,
{
    let setup = prepare(reader, interactions, clusters, pairs)?;
    Ok(observed(&setup, threshold).0)
}

/// CellPhoneDB statistical analysis.
///
/// Permutes the cluster labels across all included cells `n_perm` times and
/// counts how often the permuted interaction mean strictly exceeds the
/// observed one. Only entries with a positive observed mean and a passing
/// gate are permuted; the rest get p = 1, as in CellPhoneDB. The rayon tasks
/// are permutation batches, each seeded from `(seed, batch)`, so results do
/// not depend on the thread count.
///
/// ### Params
///
/// * `reader` - Gene-based reader. Uses the normalised layer.
/// * `interactions` - The interactions, resolved to gene indices.
/// * `clusters` - Disjoint cell indices per cluster. Cells in no cluster are
///   ignored, both for the statistics and for the permutation.
/// * `pairs` - Cluster pairs to test. `None` tests all ordered pairs.
/// * `params` - [`CellPhoneDbParams`]. `None` uses the defaults.
/// * `verbose` - `0` silent, `1` normal, `2` detailed. Normal reports the
///   permutations in 10% steps.
///
/// ### Returns
///
/// The [`CellPhoneDbRes`].
///
/// ### References
///
/// Efremova et al., Nat Protoc, 2020; Troulé et al., Nat Protoc, 2025.
pub fn cellphonedb_statistical<T, S>(
    reader: &S,
    interactions: &[LrInteraction],
    clusters: &[Vec<usize>],
    pairs: Option<&[(usize, usize)]>,
    params: Option<CellPhoneDbParams<T>>,
    verbose: usize,
) -> Result<CellPhoneDbRes<T>, BixverseErrors>
where
    T: BixverseFloat + Send + Sync,
    S: SingleCellReading,
{
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();
    let params = params.unwrap_or_default();
    if params.n_perm == 0 {
        return Err(BixverseErrors::MustBePositive("n_perm".into()));
    }

    let setup = prepare(reader, interactions, clusters, pairs)?;
    let (obs, real) = observed(&setup, params.threshold);

    let n_genes = setup.genes.len();
    let n_clusters = setup.sizes.len();
    let n_pairs = setup.pairs.len();
    let n_cells = setup.labels.len();

    let active: Vec<usize> = (0..real.len())
        .filter(|&e| real[e] > 0.0 && obs.gate[e])
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "Running CellPhoneDB: {} interactions x {} pairs ({} tested), {} permutations",
            setup.rows_a.len().separate_with_underscores(),
            n_pairs.separate_with_underscores(),
            active.len().separate_with_underscores(),
            params.n_perm.separate_with_underscores()
        );
    }

    let batch = resolve_perm_batch(params.perm_batch, params.n_perm);
    let n_batches = params.n_perm.div_ceil(batch);
    let start_perm = Instant::now();
    let perm_done = AtomicUsize::new(0);

    let counts: Vec<u32> = (0..n_batches)
        .into_par_iter()
        .fold(
            || vec![0u32; active.len()],
            |mut counts, bi| {
                let b = batch.min(params.n_perm - bi * batch);
                let mut rng = SmallRng::seed_from_u64(tree_seed(params.seed, bi));
                let mut perm = setup.labels.clone();
                let mut labels = vec![0u16; n_cells * b];
                for j in 0..b {
                    perm.copy_from_slice(&setup.labels);
                    perm.shuffle(&mut rng);
                    for (c, &k) in perm.iter().enumerate() {
                        labels[c * b + j] = k;
                    }
                }

                let mut means = vec![0.0; n_genes * n_clusters * b];
                group_means_batched(&setup.store, &labels, b, &setup.sizes, &mut means);

                for (cnt, &e) in counts.iter_mut().zip(active.iter()) {
                    let (i, p) = (e / n_pairs, e % n_pairs);
                    let (a, bb) = setup.pairs[p];
                    for j in 0..b {
                        let x = partner_mean(&means, &setup.rows_a[i], a, j, n_clusters, b);
                        let y = partner_mean(&means, &setup.rows_b[i], bb, j, n_clusters, b);
                        if interaction_mean(x, y) > real[e] {
                            *cnt += 1;
                        }
                    }
                }

                if verbosity.normal_verbosity() {
                    let done = perm_done.fetch_add(b, Ordering::Relaxed) + b;
                    report_decile_progress(
                        done,
                        done - b,
                        params.n_perm,
                        "permutations",
                        start_perm.elapsed(),
                    );
                }

                counts
            },
        )
        .reduce(
            || vec![0u32; active.len()],
            |mut acc, other| {
                acc.iter_mut().zip(other).for_each(|(a, o)| *a += o);
                acc
            },
        );

    let mut pvals = Mat::from_fn(real.len() / n_pairs.max(1), n_pairs, |_, _| T::one());
    let n_perm = T::from_usize(params.n_perm).unwrap();
    for (&cnt, &e) in counts.iter().zip(active.iter()) {
        pvals[(e / n_pairs, e % n_pairs)] = T::from_u32(cnt).unwrap() / n_perm;
    }

    if verbosity.normal_verbosity() {
        println!("CellPhoneDB finished in {:.2?}", start.elapsed());
    }

    Ok(CellPhoneDbRes { obs, pvals })
}

/// CellPhoneDB DEG gate.
///
/// An interaction is relevant on `(A, B)` if it passes the expression gate
/// and partner a is differentially expressed in `A` or partner b in `B`. A
/// complex counts as differentially expressed if any subunit is.
///
/// ### Params
///
/// * `interactions` - The interactions passed to [`cellphonedb_observed`].
/// * `deg_genes` - Differentially expressed gene indices per cluster, in the
///   cluster order used for the observed statistics.
/// * `obs` - The observed statistics.
///
/// ### Returns
///
/// The relevance mask, row-major `(interactions, pairs)`.
pub fn cpdb_deg_gate<T>(
    interactions: &[LrInteraction],
    deg_genes: &[Vec<usize>],
    obs: &CellPhoneDbObs<T>,
) -> Result<Vec<bool>, BixverseErrors> {
    let n_clusters = obs.gene_mean.ncols();
    if deg_genes.len() != n_clusters {
        return Err(BixverseErrors::InvalidArgument(format!(
            "CellPhoneDB: {} DEG sets for {n_clusters} clusters.",
            deg_genes.len()
        )));
    }
    let n_pairs = obs.pairs.len();
    if obs.gate.len() != interactions.len() * n_pairs {
        return Err(BixverseErrors::InvalidArgument(format!(
            "CellPhoneDB: {} interactions do not match the observed statistics.",
            interactions.len()
        )));
    }

    let deg: Vec<FxHashSet<usize>> = deg_genes
        .iter()
        .map(|g| g.iter().copied().collect())
        .collect();
    let is_deg = |subunits: &[usize], k: usize| subunits.iter().any(|g| deg[k].contains(g));

    let mut relevant = vec![false; obs.gate.len()];
    for (i, inter) in interactions.iter().enumerate() {
        for (p, &(a, b)) in obs.pairs.iter().enumerate() {
            let e = i * n_pairs + p;
            relevant[e] =
                obs.gate[e] && (is_deg(&inter.partner_a, a) || is_deg(&inter.partner_b, b));
        }
    }
    Ok(relevant)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_cell::sc_data::in_memory_io::InMemorySparseReader;
    use crate::single_cell::sc_traits::F16;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    /// CSC (cells, genes) from dense columns, norm values in `data_2`.
    fn csc_from_columns(columns: &[Vec<f32>]) -> CompressedSparseData2<u32, f32> {
        let n_cells = columns[0].len();
        let mut data_2: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];
        for col in columns {
            for (i, &v) in col.iter().enumerate() {
                if v != 0.0 {
                    data_2.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(data_2.len() as u32);
        }
        let data: Vec<u32> = data_2.iter().map(|&v| (v * 100.0) as u32).collect();
        CompressedSparseData2::new_csc(
            &data,
            &indices,
            &indptr,
            Some(&data_2),
            (n_cells, columns.len()),
        )
    }

    fn contiguous_clusters(sizes: &[usize]) -> Vec<Vec<usize>> {
        let mut start = 0;
        sizes
            .iter()
            .map(|&n| {
                let c: Vec<usize> = (start..start + n).collect();
                start += n;
                c
            })
            .collect()
    }

    fn random_columns(n_genes: usize, n_cells: usize, density: f64, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n_genes)
            .map(|_| {
                (0..n_cells)
                    .map(|_| {
                        if rng.random::<f64>() < density {
                            rng.random::<f32>() * 3.0 + 0.1
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn q(v: f32) -> f64 {
        F16::from_f32(v).to_f32() as f64
    }

    #[test]
    fn test_cpdb_complex_min_and_zero_rule() {
        // 4 cells, 2 clusters of 2. Gene 0, 1 form a complex; gene 2 is a
        // single partner, zero in cluster 1.
        let columns = vec![
            vec![1.0, 1.0, 2.0, 2.0],
            vec![0.5, 0.5, 4.0, 4.0],
            vec![3.0, 1.0, 0.0, 0.0],
        ];
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[2, 2]);
        let inter = vec![LrInteraction {
            partner_a: vec![0, 1],
            partner_b: vec![2],
        }];
        let obs = cellphonedb_observed::<f64, _>(&reader, &inter, &clusters, None, 0.1).unwrap();

        assert_eq!(obs.pairs, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        // complex: min(1, 0.5) = 0.5 in c0, min(2, 4) = 2 in c1. gene 2: 2 in c0.
        assert_relative_eq!(obs.means[(0, 0)], (0.5 + 2.0) / 2.0, epsilon = 1e-3);
        assert_eq!(obs.means[(0, 1)], 0.0);
        assert_relative_eq!(obs.means[(0, 2)], (2.0 + 2.0) / 2.0, epsilon = 1e-3);
        assert_eq!(obs.means[(0, 3)], 0.0);
        assert_eq!(obs.gate, vec![true, false, true, false]);
    }

    #[test]
    fn test_cpdb_gate_is_strict() {
        // 10 cells in one cluster, gene expressed in exactly one: pct = 0.1.
        let mut col = vec![0.0; 10];
        col[3] = 1.0;
        let matrix = csc_from_columns(&[col.clone(), col]);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[10]);
        let inter = vec![LrInteraction {
            partner_a: vec![0],
            partner_b: vec![1],
        }];
        let at = cellphonedb_observed::<f64, _>(&reader, &inter, &clusters, None, 0.1).unwrap();
        assert!(!at.gate[0]);
        let below = cellphonedb_observed::<f64, _>(&reader, &inter, &clusters, None, 0.09).unwrap();
        assert!(below.gate[0]);
    }

    #[test]
    fn test_cpdb_single_cluster_p_is_zero() {
        // With one cluster every permutation reproduces the observed mean,
        // and a strict > never fires.
        let columns = random_columns(2, 50, 0.6, 1);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[50]);
        let inter = vec![LrInteraction {
            partner_a: vec![0],
            partner_b: vec![1],
        }];
        let params = CellPhoneDbParams::new(100, 0.1, 3, None);
        let res =
            cellphonedb_statistical::<f64, _>(&reader, &inter, &clusters, None, Some(params), 0)
                .unwrap();
        assert_eq!(res.pvals[(0, 0)], 0.0);
    }

    #[test]
    fn test_cpdb_planted_signal() {
        // Ligand only in cluster 0, receptor only in cluster 1.
        let n = 40;
        let mut lig = vec![0.0; 3 * n];
        let mut rec = vec![0.0; 3 * n];
        let mut rng = StdRng::seed_from_u64(9);
        for c in 0..n {
            lig[c] = rng.random::<f32>() + 1.0;
            rec[n + c] = rng.random::<f32>() + 1.0;
        }
        let matrix = csc_from_columns(&[lig, rec]);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[n, n, n]);
        let inter = vec![LrInteraction {
            partner_a: vec![0],
            partner_b: vec![1],
        }];
        let params = CellPhoneDbParams::new(200, 0.1, 5, Some(7));
        let res =
            cellphonedb_statistical::<f64, _>(&reader, &inter, &clusters, None, Some(params), 0)
                .unwrap();
        for (p, &(a, b)) in res.obs.pairs.iter().enumerate() {
            let want = if (a, b) == (0, 1) { 0.0 } else { 1.0 };
            assert_eq!(res.pvals[(0, p)], want, "pair ({a}, {b})");
        }
    }

    #[test]
    fn test_cpdb_reproducible_and_thread_independent() {
        let columns = random_columns(6, 300, 0.3, 11);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[90, 60, 100, 50]);
        let inter = vec![
            LrInteraction {
                partner_a: vec![0],
                partner_b: vec![1, 2],
            },
            LrInteraction {
                partner_a: vec![3],
                partner_b: vec![4],
            },
            LrInteraction {
                partner_a: vec![5, 0],
                partner_b: vec![5],
            },
        ];
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let params = CellPhoneDbParams::new(250, 0.1, 17, None);
                cellphonedb_statistical::<f64, _>(&reader, &inter, &clusters, None, Some(params), 0)
                    .unwrap()
                    .pvals
            })
        };
        let p1 = run(1);
        let p4 = run(4);
        let p4b = run(4);
        assert_eq!(p1, p4);
        assert_eq!(p4, p4b);
        assert!(
            p1.col_iter()
                .flat_map(|c| c.iter().copied().collect::<Vec<_>>())
                .any(|v| v > 0.0 && v < 1.0)
        );
    }

    #[test]
    fn test_cpdb_means_match_dense_reference() {
        let columns = random_columns(3, 120, 0.4, 21);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let sizes = [30, 50, 40];
        let clusters = contiguous_clusters(&sizes);
        let inter = vec![LrInteraction {
            partner_a: vec![0, 1],
            partner_b: vec![2],
        }];
        let obs = cellphonedb_observed::<f64, _>(&reader, &inter, &clusters, None, 0.1).unwrap();
        let mean = |g: usize, k: usize| {
            clusters[k].iter().map(|&c| q(columns[g][c])).sum::<f64>() / sizes[k] as f64
        };
        for (p, &(a, b)) in obs.pairs.iter().enumerate() {
            let x = mean(0, a).min(mean(1, a));
            let y = mean(2, b);
            assert_relative_eq!(obs.means[(0, p)], interaction_mean(x, y), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cpdb_deg_gate() {
        let columns = random_columns(3, 60, 0.8, 4);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let clusters = contiguous_clusters(&[30, 30]);
        let inter = vec![LrInteraction {
            partner_a: vec![0, 1],
            partner_b: vec![2],
        }];
        let obs = cellphonedb_observed::<f64, _>(&reader, &inter, &clusters, None, 0.1).unwrap();
        assert!(obs.gate.iter().all(|&g| g));
        // subunit 1 is a DEG in cluster 0, nothing in cluster 1
        let rel = cpdb_deg_gate(&inter, &[vec![1], vec![]], &obs).unwrap();
        // pairs (0,0), (0,1), (1,0), (1,1)
        assert_eq!(rel, vec![true, true, false, false]);
    }

    #[test]
    fn test_cpdb_rejects_bad_inputs() {
        let columns = random_columns(2, 10, 0.5, 2);
        let matrix = csc_from_columns(&columns);
        let reader = InMemorySparseReader::new(&matrix, None).unwrap();
        let inter = |a: Vec<usize>| {
            vec![LrInteraction {
                partner_a: a,
                partner_b: vec![1],
            }]
        };
        let ok = contiguous_clusters(&[5, 5]);
        let overlap = vec![vec![0, 1, 2], vec![2, 3]];
        assert!(matches!(
            cellphonedb_observed::<f64, _>(&reader, &inter(vec![5]), &ok, None, 0.1),
            Err(BixverseErrors::CpdbGeneIndexOutOfRange { .. })
        ));
        assert!(matches!(
            cellphonedb_observed::<f64, _>(&reader, &inter(vec![]), &ok, None, 0.1),
            Err(BixverseErrors::CpdbEmptyPartner { .. })
        ));
        assert!(matches!(
            cellphonedb_observed::<f64, _>(&reader, &inter(vec![0]), &overlap, None, 0.1),
            Err(BixverseErrors::CpdbClusterOverlap { cell: 2 })
        ));
    }
}
