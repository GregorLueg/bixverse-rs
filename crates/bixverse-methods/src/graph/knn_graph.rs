//! kNN graph construction: one [`KnnSymmetrisation`] choice turns a directed
//! kNN into an undirected weighted [`SparseGraph`].
//!
//! Every variant takes the neighbours row-major (`k` per node, self excluded,
//! sorted by distance) and builds the symmetric CSR directly, rows sorted by
//! column, without going through an edge-pair list.

use num_traits::Float;
use rayon::prelude::*;
use std::iter::Sum;
use std::time::Instant;

use crate::graph::graph_structures::UnionFind;
use crate::prelude::*;
use bixverse_core::core::math::sparse::{CompressedSparseData2, CompressedSparseFormat};

///////////
// Const //
///////////

/// Components below this size after pruning get reconnected by the sNN
/// rescue.
const MIN_COMPONENT_SIZE: usize = 10;

/// Rows per rayon task in the sNN pass. Each task owns one `n`-long score
/// buffer, so this bounds how often that buffer is allocated rather than how
/// the work is balanced.
const SNN_ROW_CHUNK: usize = 2048;

/// umap-learn's `SMOOTH_K_TOLERANCE`: binary-search tolerance and the
/// interpolation cut-off for fractional `local_connectivity`.
const SMOOTH_K_TOLERANCE: f64 = 1e-5;

/// umap-learn's `MIN_K_DIST_SCALE`: floor on sigma as a fraction of the mean
/// neighbour distance.
const MIN_K_DIST_SCALE: f64 = 1e-3;

/// Binary-search iterations for the UMAP bandwidth, umap-learn's `n_iter`.
const SMOOTH_K_ITERS: usize = 64;

/// Default sNN prune. For the Jaccard weight a threshold `p` keeps pairs that
/// share at least a fraction `2p / (1 + p)` of their self-inclusive
/// neighbourhoods, independent of `k`. `1/16` asks for about 12%, in line with
/// Seurat's `1/15`, and sits off the discrete weight `1/15` takes at `k = 15`,
/// where float rounding would decide whether a 2-shared edge survives.
pub const SNN_PRUNE_DEFAULT: f64 = 1.0 / 16.0;

/// Floor on the rank sNN weight before scaling, as in bluster.
const SNN_RANK_FLOOR: f64 = 1e-6;

///////////
// Enums //
///////////

/// Similarity used to weight sNN edges.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SnnSimilarity {
    /// Jaccard index of the self-inclusive neighbour sets
    #[default]
    Jaccard,
    /// bluster's rank weight: `k - r / 2`, `r` the smallest combined rank of
    /// a shared neighbour, scaled to `(0, 1]`
    Rank,
}

/// How a directed kNN becomes an undirected graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnnSymmetrisation {
    /// kNN union. Mutual pairs weigh 2 and one-sided pairs 1, unless `binary`
    /// sets every weight to 1.
    Knn {
        /// Set every weight to 1
        binary: bool,
    },
    /// Shared nearest neighbours: an edge between every pair of nodes that
    /// shares any neighbour (bluster / Seurat).
    Snn {
        /// Edge weight
        similarity: SnnSimilarity,
        /// Edges below this weight are dropped
        prune: f64,
        /// Reconnect components that pruning leaves below
        /// [`MIN_COMPONENT_SIZE`] nodes
        rescue: bool,
    },
    /// UMAP fuzzy simplicial set (the scanpy default). Needs distances.
    Umap {
        /// Number of neighbours assumed at distance zero, umap-learn's
        /// `local_connectivity`
        local_connectivity: f64,
        /// Fuzzy union (1) versus fuzzy intersection (0), umap-learn's
        /// `set_op_mix_ratio`
        mix_ratio: f64,
    },
}

/// Defaults to the bluster sNN with Jaccard weights at a prune of
/// [`SNN_PRUNE_DEFAULT`]
impl Default for KnnSymmetrisation {
    fn default() -> Self {
        KnnSymmetrisation::Snn {
            similarity: SnnSimilarity::Jaccard,
            prune: SNN_PRUNE_DEFAULT,
            rescue: true,
        }
    }
}

/// Parse an sNN similarity name.
///
/// ### Params
///
/// * `s` - `"jaccard"` or `"rank"`, case-insensitive
///
/// ### Returns
///
/// The [`SnnSimilarity`], or `None` for an unknown name
pub fn parse_snn_similarity(s: &str) -> Option<SnnSimilarity> {
    match s.to_lowercase().as_str() {
        "jaccard" => Some(SnnSimilarity::Jaccard),
        "rank" => Some(SnnSimilarity::Rank),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Cast an `f64` constant or count into the working float type.
///
/// ### Params
///
/// * `x` - The value
///
/// ### Returns
///
/// `x` as `T`
#[inline(always)]
fn t<T: BixverseFloat>(x: f64) -> T {
    T::from_f64(x).expect("f64 fits every float type")
}

/// Check the shape and range of a row-major kNN.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `n_dist` - Length of the distances, if given
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
///
/// ### Returns
///
/// `Ok(())` or the first violation found
fn check_knn(idx: &[u32], n_dist: Option<usize>, k: usize, n: usize) -> Result<(), MethodsErrors> {
    if idx.len() != n * k {
        return Err(MethodsErrors::GraphLengthMismatch {
            name: "idx",
            expected: n * k,
            got: idx.len(),
        });
    }
    if let Some(got) = n_dist
        && got != n * k
    {
        return Err(MethodsErrors::GraphLengthMismatch {
            name: "dist",
            expected: n * k,
            got,
        });
    }
    if let Some(&bad) = idx.par_iter().find_any(|&&j| j as usize >= n) {
        return Err(MethodsErrors::GraphIndexOutOfRange {
            name: "idx",
            index: bad as usize,
            n_nodes: n,
        });
    }
    Ok(())
}

/// Wrap a symmetric, row-sorted CSR into an undirected [`SparseGraph`].
///
/// ### Params
///
/// * `indptr` - Row offsets, `n + 1`
/// * `indices` - Column indices
/// * `data` - Edge weights
/// * `n` - Number of nodes
///
/// ### Returns
///
/// The undirected graph
fn csr_to_graph<T>(indptr: Vec<usize>, indices: Vec<u32>, data: Vec<T>, n: usize) -> SparseGraph<T>
where
    T: BixverseFloat + Sum,
{
    let adjacency = CompressedSparseData2 {
        data,
        indices,
        indptr: indptr.into_iter().map(|p| p as u32).collect(),
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (n, n),
    };
    SparseGraph::new(n, adjacency, false)
}

/// Exclusive prefix sum of per-row lengths into CSR offsets.
///
/// ### Params
///
/// * `lens` - Entries per row
///
/// ### Returns
///
/// Offsets of length `lens.len() + 1`
fn offsets(lens: &[usize]) -> Vec<usize> {
    let mut ptr = Vec::with_capacity(lens.len() + 1);
    ptr.push(0);
    let mut acc = 0;
    for &l in lens {
        acc += l;
        ptr.push(acc);
    }
    ptr
}

/// Hand out one disjoint mutable slice per CSR row.
///
/// ### Params
///
/// * `data` - The flat CSR array
/// * `ptr` - Row offsets into it
///
/// ### Returns
///
/// One slice per row, in row order
fn row_slices_mut<'a, T>(mut data: &'a mut [T], ptr: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(ptr.len().saturating_sub(1));
    for w in ptr.windows(2) {
        let (row, rest) = data.split_at_mut(w[1] - w[0]);
        out.push(row);
        data = rest;
    }
    out
}

/// Flatten sorted per-row adjacency lists into a graph.
///
/// ### Params
///
/// * `rows` - Per-node `(column, weight)` lists, sorted by column, already
///   symmetric
/// * `n` - Number of nodes
///
/// ### Returns
///
/// The undirected graph
fn lists_to_graph<T>(rows: &[Vec<(u32, T)>], n: usize) -> SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + Sum,
{
    let lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    let ptr = offsets(&lens);
    let mut indices = vec![0u32; ptr[n]];
    let mut data = vec![T::zero(); ptr[n]];
    row_slices_mut(&mut indices, &ptr)
        .into_par_iter()
        .zip(row_slices_mut(&mut data, &ptr))
        .zip(rows.par_iter())
        .for_each(|((ix, dx), row)| {
            for (e, &(c, v)) in row.iter().enumerate() {
                ix[e] = c;
                dx[e] = v;
            }
        });
    csr_to_graph(ptr, indices, data, n)
}

/// Symmetrise a weighted directed kNN.
///
/// Row `i` of the output is the sorted union of `i`'s out-neighbours and
/// in-neighbours, each weighted `combine(w_ij, w_ji)` with a missing direction
/// passed as zero. Results equal to zero are dropped, as scipy's
/// `eliminate_zeros` does. Self entries are skipped.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `w` - Directed weight per kNN entry, same layout
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `combine` - Applied to `(w_ij, w_ji)`
///
/// ### Returns
///
/// The undirected graph
fn symmetrise<T, F>(idx: &[u32], w: &[T], k: usize, n: usize, combine: F) -> SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + Sum,
    F: Fn(T, T) -> T + Sync,
{
    // out rows sorted by column; self entries removed
    let out: Vec<Vec<(u32, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row: Vec<(u32, T)> = idx[i * k..(i + 1) * k]
                .iter()
                .zip(&w[i * k..(i + 1) * k])
                .filter(|&(&j, _)| j as usize != i)
                .map(|(&j, &wt)| (j, wt))
                .collect();
            row.sort_unstable_by_key(|e| e.0);
            row
        })
        .collect();

    // in rows via counting sort; filling sources in ascending order keeps
    // every in row sorted by column
    let mut in_len = vec![0usize; n];
    for row in &out {
        for &(j, _) in row {
            in_len[j as usize] += 1;
        }
    }
    let in_ptr = offsets(&in_len);
    let mut in_entries = vec![(0u32, T::zero()); in_ptr[n]];
    let mut cursor = in_ptr.clone();
    for (i, row) in out.iter().enumerate() {
        for &(j, wt) in row {
            in_entries[cursor[j as usize]] = (i as u32, wt);
            cursor[j as usize] += 1;
        }
    }

    // merge each out row with its in row
    let merged: Vec<Vec<(u32, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let a = &out[i];
            let b = &in_entries[in_ptr[i]..in_ptr[i + 1]];
            let mut row = Vec::with_capacity(a.len() + b.len());
            let (mut p, mut q) = (0, 0);
            while p < a.len() || q < b.len() {
                let (col, wij, wji) = match (a.get(p), b.get(q)) {
                    (Some(&(ca, wa)), Some(&(cb, wb))) if ca == cb => {
                        p += 1;
                        q += 1;
                        (ca, wa, wb)
                    }
                    (Some(&(ca, wa)), Some(&(cb, _))) if ca < cb => {
                        p += 1;
                        (ca, wa, T::zero())
                    }
                    (Some(&(ca, wa)), None) => {
                        p += 1;
                        (ca, wa, T::zero())
                    }
                    (_, Some(&(cb, wb))) => {
                        q += 1;
                        (cb, T::zero(), wb)
                    }
                    (None, None) => unreachable!(),
                };
                let v = combine(wij, wji);
                if v != T::zero() {
                    row.push((col, v));
                }
            }
            row
        })
        .collect();

    lists_to_graph(&merged, n)
}

/////////
// kNN //
/////////

/// kNN union.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `binary` - Set every weight to 1 instead of counting the directions
///
/// ### Returns
///
/// The undirected graph
fn knn_union<T>(idx: &[u32], k: usize, n: usize, binary: bool) -> SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + Sum,
{
    let ones = vec![T::one(); idx.len()];
    if binary {
        symmetrise(idx, &ones, k, n, |_, _| T::one())
    } else {
        symmetrise(idx, &ones, k, n, |a, b| a + b)
    }
}

/////////
// sNN //
/////////

/// sNN weight from the shared-member count and the smallest combined rank.
///
/// ### Params
///
/// * `count` - Shared members of the two self-inclusive neighbour sets
/// * `min_rank` - Smallest `rank_a + rank_b` over the shared members
/// * `k` - Neighbours per node
/// * `similarity` - Which weight
///
/// ### Returns
///
/// The edge weight
#[inline]
fn snn_weight<T: BixverseFloat>(
    count: u32,
    min_rank: u32,
    k: usize,
    similarity: SnnSimilarity,
) -> T {
    let kf: T = t(k as f64);
    match similarity {
        SnnSimilarity::Jaccard => {
            let c: T = t(count as f64);
            c / (t::<T>(2.0) * (kf + T::one()) - c)
        }
        SnnSimilarity::Rank => {
            let raw = kf - t::<T>(min_rank as f64) / t::<T>(2.0);
            let floor: T = t(SNN_RANK_FLOOR);
            (if raw > floor { raw } else { floor }) / kf
        }
    }
}

/// sNN weight between two nodes, built from their kNN rows on the fly.
///
/// Only meant for the handful of pairs the component rescue looks at.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `k` - Neighbours per node
/// * `a` - First node
/// * `b` - Second node
/// * `similarity` - Which weight
///
/// ### Returns
///
/// The weight the full sNN pass would assign the pair
fn pair_snn_weight<T: BixverseFloat>(
    idx: &[u32],
    k: usize,
    a: usize,
    b: usize,
    similarity: SnnSimilarity,
) -> T {
    let ranked = |c: usize| -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = std::iter::once((c as u32, 0))
            .chain(
                idx[c * k..(c + 1) * k]
                    .iter()
                    .enumerate()
                    .map(|(r, &nb)| (nb, r as u32 + 1)),
            )
            .collect();
        v.sort_unstable();
        v
    };
    let (sa, sb) = (ranked(a), ranked(b));
    let (mut pa, mut pb) = (0, 0);
    let mut count = 0u32;
    let mut min_rank = 2 * k as u32;
    while pa < sa.len() && pb < sb.len() {
        match sa[pa].0.cmp(&sb[pb].0) {
            std::cmp::Ordering::Equal => {
                count += 1;
                min_rank = min_rank.min(sa[pa].1 + sb[pb].1);
                pa += 1;
                pb += 1;
            }
            std::cmp::Ordering::Less => pa += 1,
            std::cmp::Ordering::Greater => pb += 1,
        }
    }
    snn_weight(count, min_rank, k, similarity)
}

/// Reconnect the small components that pruning cut off.
///
/// Leiden turns every disconnected component into its own cluster, and on
/// large data sets pruning leaves hundreds of them, mostly pairs of nodes that
/// are each other's strongest neighbour. Every component below
/// [`MIN_COMPONENT_SIZE`] gets its strongest kNN edge to a node outside it,
/// repeated until none is left or the remaining ones have no kNN edge leaving
/// them. A kNN pair always shares at least the partner itself, so the edge
/// exists in the unpruned sNN.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `similarity` - Which weight
/// * `lower` - Per-node lists of `(smaller neighbour, weight)`; rescue edges
///   are inserted, keeping each list sorted
///
/// ### Returns
///
/// Number of rescue edges added
fn rescue_small_components<T>(
    idx: &[u32],
    k: usize,
    n: usize,
    similarity: SnnSimilarity,
    lower: &mut [Vec<(u32, T)>],
) -> usize
where
    T: BixverseFloat + Send + Sync,
{
    if k == 0 {
        return 0;
    }
    let mut uf = UnionFind::new(n);
    for (j, row) in lower.iter().enumerate() {
        for &(i, _) in row {
            uf.union(i, j as u32);
        }
    }

    let mut n_rescued = 0;
    loop {
        let roots: Vec<u32> = (0..n as u32).map(|i| uf.find(i)).collect();
        let mut size = vec![0usize; n];
        for &r in &roots {
            size[r as usize] += 1;
        }

        // (root, weight, node, partner); ties go to the closer neighbour
        let mut candidates: Vec<(u32, T, usize, usize)> = (0..n)
            .into_par_iter()
            .filter(|&i| size[roots[i] as usize] < MIN_COMPONENT_SIZE)
            .filter_map(|i| {
                idx[i * k..(i + 1) * k]
                    .iter()
                    .map(|&nb| nb as usize)
                    .filter(|&nb| roots[nb] != roots[i])
                    .map(|nb| (pair_snn_weight::<T>(idx, k, i, nb, similarity), nb))
                    .fold(None, |acc: Option<(T, usize)>, cur| match acc {
                        Some(a) if a.0 >= cur.0 => Some(a),
                        _ => Some(cur),
                    })
                    .map(|(w, nb)| (roots[i], w, i, nb))
            })
            .collect();
        candidates.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.2.cmp(&b.2))
        });
        candidates.dedup_by_key(|c| c.0);

        let before = n_rescued;
        for (_, w, i, nb) in candidates {
            // two components can pick each other in the same round
            if uf.union(i as u32, nb as u32) {
                let (lo, hi) = (i.min(nb), i.max(nb));
                let row = &mut lower[hi];
                let pos = row.partition_point(|e| (e.0 as usize) < lo);
                row.insert(pos, (lo as u32, w));
                n_rescued += 1;
            }
        }
        if n_rescued == before {
            break;
        }
    }
    n_rescued
}

/// Full sNN graph.
///
/// For each node `j`, every node `i < j` that shares a member of the
/// self-inclusive neighbour sets is found through the reverse kNN index, and
/// the shared count (Jaccard) or the smallest combined rank (Rank) is
/// accumulated in a per-task `n`-long buffer. The lower-triangle lists that
/// come out are sorted, so the symmetric CSR is assembled in one scatter with
/// no per-row sort.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `similarity` - Which weight
/// * `prune` - Edges below this weight are dropped
/// * `rescue` - Reconnect small components afterwards
///
/// ### Returns
///
/// `(graph, number of rescue edges)`
fn snn_full<T>(
    idx: &[u32],
    k: usize,
    n: usize,
    similarity: SnnSimilarity,
    prune: T,
    rescue: bool,
) -> (SparseGraph<T>, usize)
where
    T: BixverseFloat + Send + Sync + Sum,
{
    // reverse index: for each node v, the (i, rank) with v in N+(i), filled
    // in i order with i's own entry ahead of its neighbour entries
    let mut rev_len = vec![1usize; n];
    for &nb in idx {
        rev_len[nb as usize] += 1;
    }
    let rev_ptr = offsets(&rev_len);
    let mut rev = vec![(0u32, 0u32); rev_ptr[n]];
    let mut cursor = rev_ptr.clone();
    for i in 0..n {
        rev[cursor[i]] = (i as u32, 0);
        cursor[i] += 1;
        for (r, &nb) in idx[i * k..(i + 1) * k].iter().enumerate() {
            let nb = nb as usize;
            rev[cursor[nb]] = (i as u32, r as u32 + 1);
            cursor[nb] += 1;
        }
    }

    // u32::MAX marks an untouched slot in both modes
    let mut lower: Vec<Vec<(u32, T)>> = vec![Vec::new(); n];
    lower
        .par_chunks_mut(SNN_ROW_CHUNK)
        .enumerate()
        .for_each(|(c, chunk)| {
            let mut scores = vec![u32::MAX; n];
            let mut added: Vec<u32> = Vec::with_capacity(256);
            for (off, out) in chunk.iter_mut().enumerate() {
                let j = c * SNN_ROW_CHUNK + off;
                added.clear();
                for r in 0..=k {
                    let v = if r == 0 {
                        j
                    } else {
                        idx[j * k + r - 1] as usize
                    };
                    for &(o, o_rank) in &rev[rev_ptr[v]..rev_ptr[v + 1]] {
                        if (o as usize) >= j {
                            continue;
                        }
                        let s = &mut scores[o as usize];
                        match similarity {
                            SnnSimilarity::Jaccard => {
                                if *s == u32::MAX {
                                    *s = 1;
                                    added.push(o);
                                } else {
                                    *s += 1;
                                }
                            }
                            SnnSimilarity::Rank => {
                                let comb = r as u32 + o_rank;
                                if *s == u32::MAX {
                                    *s = comb;
                                    added.push(o);
                                } else if comb < *s {
                                    *s = comb;
                                }
                            }
                        }
                    }
                }
                added.sort_unstable();
                out.extend(added.iter().filter_map(|&o| {
                    let s = scores[o as usize];
                    let w: T = match similarity {
                        SnnSimilarity::Jaccard => snn_weight(s, 0, k, similarity),
                        SnnSimilarity::Rank => snn_weight(0, s, k, similarity),
                    };
                    (w >= prune).then_some((o, w))
                }));
                for &o in &added {
                    scores[o as usize] = u32::MAX;
                }
            }
        });

    let n_rescued = if rescue {
        rescue_small_components(idx, k, n, similarity, &mut lower)
    } else {
        0
    };

    (lower_to_graph(&lower, n), n_rescued)
}

/// Mirror sorted lower-triangle lists into a symmetric CSR.
///
/// Row `r` holds its own lower list followed by the `j > r` that listed it,
/// in ascending `j`, so every row comes out sorted.
///
/// ### Params
///
/// * `lower` - Per-node lists of `(smaller neighbour, weight)`, each sorted
/// * `n` - Number of nodes
///
/// ### Returns
///
/// The undirected graph, rows sorted by column
fn lower_to_graph<T>(lower: &[Vec<(u32, T)>], n: usize) -> SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + Sum,
{
    let mut lens: Vec<usize> = lower.iter().map(|r| r.len()).collect();
    for row in lower {
        for &(i, _) in row {
            lens[i as usize] += 1;
        }
    }
    let ptr = offsets(&lens);
    let mut indices = vec![0u32; ptr[n]];
    let mut data = vec![T::zero(); ptr[n]];

    // own lower entries first, in parallel
    let mut cursor: Vec<usize> = (0..n).map(|r| ptr[r] + lower[r].len()).collect();
    row_slices_mut(&mut indices, &ptr)
        .into_par_iter()
        .zip(row_slices_mut(&mut data, &ptr))
        .zip(lower.par_iter())
        .for_each(|((ix, dx), row)| {
            for (e, &(c, v)) in row.iter().enumerate() {
                ix[e] = c;
                dx[e] = v;
            }
        });
    // then the upper half, ascending j keeps each row sorted
    for (j, row) in lower.iter().enumerate() {
        for &(i, v) in row {
            let p = &mut cursor[i as usize];
            indices[*p] = j as u32;
            data[*p] = v;
            *p += 1;
        }
    }
    csr_to_graph(ptr, indices, data, n)
}

//////////
// UMAP //
//////////

/// Per-node UMAP bandwidth and local-connectivity distance.
///
/// Mirrors umap-learn's `smooth_knn_dist`, whose neighbour rows include the
/// point itself at distance zero. Here self is excluded, so the target is
/// `log2(k + 1)` and the means divide by `k + 1` to match.
///
/// ### Params
///
/// * `dist` - Distances, row-major, `k` per node, ascending
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `local_connectivity` - Neighbours assumed at distance zero
///
/// ### Returns
///
/// `(sigmas, rhos)`
fn smooth_knn_dist<T>(dist: &[T], k: usize, n: usize, local_connectivity: T) -> (Vec<T>, Vec<T>)
where
    T: BixverseFloat + Send + Sync + Sum,
{
    let target: T = t::<T>((k + 1) as f64).log2();
    let tol: T = t(SMOOTH_K_TOLERANCE);
    let scale: T = t(MIN_K_DIST_SCALE);
    let two: T = t(2.0);
    let kp1: T = t((k + 1) as f64);
    let mean_all: T = t(dist
        .par_iter()
        .map(|d| d.to_f64().unwrap_or(0.0))
        .sum::<f64>()
        / (n * (k + 1)) as f64);
    let index = Float::floor(local_connectivity).to_usize().unwrap_or(0);
    let interp: T = local_connectivity - t::<T>(index as f64);

    (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &dist[i * k..(i + 1) * k];
            let mut rho = T::zero();
            let non_zero: Vec<T> = row.iter().copied().filter(|&d| d > T::zero()).collect();
            if t::<T>(non_zero.len() as f64) >= local_connectivity {
                if index > 0 {
                    rho = non_zero[index - 1];
                    if interp > tol {
                        rho += interp * (non_zero[index] - non_zero[index - 1]);
                    }
                } else {
                    rho = interp * non_zero[0];
                }
            } else if let Some(m) = non_zero
                .iter()
                .copied()
                .reduce(|a, b| if a > b { a } else { b })
            {
                rho = m;
            }

            let (mut lo, mut hi, mut mid) = (T::zero(), T::max_value(), T::one());
            for _ in 0..SMOOTH_K_ITERS {
                let psum: T = row
                    .iter()
                    .map(|&d| {
                        let d = d - rho;
                        if d > T::zero() {
                            (-(d / mid)).exp()
                        } else {
                            T::one()
                        }
                    })
                    .sum();
                if (psum - target).abs() < tol {
                    break;
                }
                if psum > target {
                    hi = mid;
                    mid = (lo + hi) / two;
                } else {
                    lo = mid;
                    if hi >= T::max_value() {
                        mid *= two;
                    } else {
                        mid = (lo + hi) / two;
                    }
                }
            }

            let floor = if rho > T::zero() {
                scale * row.iter().copied().sum::<T>() / kp1
            } else {
                scale * mean_all
            };
            (mid.max(floor), rho)
        })
        .unzip()
}

/// UMAP fuzzy simplicial set.
///
/// Membership `exp(-(d - rho) / sigma)`, or 1 within `rho`, then
/// `mix * (A + A^T - A o A^T) + (1 - mix) * A o A^T`. With
/// `local_connectivity >= 1` every node keeps a near-1 edge to its nearest
/// neighbour, so there are no singletons.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node
/// * `dist` - Distances, same layout, ascending
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `local_connectivity` - Neighbours assumed at distance zero
/// * `mix_ratio` - Union versus intersection
///
/// ### Returns
///
/// The undirected graph
///
/// ### References
///
/// McInnes, Healy & Melville, arXiv:1802.03426, 2018
fn umap_graph<T>(
    idx: &[u32],
    dist: &[T],
    k: usize,
    n: usize,
    local_connectivity: T,
    mix_ratio: T,
) -> SparseGraph<T>
where
    T: BixverseFloat + Send + Sync + Sum,
{
    let (sigmas, rhos) = smooth_knn_dist(dist, k, n, local_connectivity);
    let mut w = vec![T::zero(); n * k];
    if k > 0 {
        w.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            for (e, slot) in row.iter_mut().enumerate() {
                let d = dist[i * k + e] - rhos[i];
                *slot = if d <= T::zero() || sigmas[i] == T::zero() {
                    T::one()
                } else {
                    (-(d / sigmas[i])).exp()
                };
            }
        });
    }
    symmetrise(idx, &w, k, n, |a, b| {
        let prod = a * b;
        mix_ratio * (a + b - prod) + (T::one() - mix_ratio) * prod
    })
}

//////////
// Main //
//////////

/// Build an undirected weighted graph from a kNN.
///
/// ### Params
///
/// * `idx` - Neighbour indices, row-major, `k` per node, self excluded
/// * `dist` - Optional distances, same layout, ascending per row. Required
///   for [`KnnSymmetrisation::Umap`]
/// * `k` - Neighbours per node
/// * `n` - Number of nodes
/// * `sym` - The [`KnnSymmetrisation`]
/// * `verbose` - `0` silent, `1` timings, `2` detailed
///
/// ### Returns
///
/// The undirected graph with a symmetric, row-sorted CSR
pub fn build_knn_graph<T>(
    idx: &[u32],
    dist: Option<&[T]>,
    k: usize,
    n: usize,
    sym: &KnnSymmetrisation,
    verbose: usize,
) -> Result<SparseGraph<T>, MethodsErrors>
where
    T: BixverseFloat + Send + Sync + Sum,
{
    check_knn(idx, dist.map(|d| d.len()), k, n)?;
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();

    let graph = match *sym {
        KnnSymmetrisation::Knn { binary } => knn_union(idx, k, n, binary),
        KnnSymmetrisation::Snn {
            similarity,
            prune,
            rescue,
        } => {
            let (g, n_rescued) = snn_full(idx, k, n, similarity, t(prune), rescue);
            if verbosity.normal_verbosity() && rescue {
                println!("Added {n_rescued} edges to reconnect components that pruning cut off.");
            }
            g
        }
        KnnSymmetrisation::Umap {
            local_connectivity,
            mix_ratio,
        } => {
            let d = dist.ok_or(MethodsErrors::KnnGraphNeedsDistances)?;
            umap_graph(idx, d, k, n, t(local_connectivity), t(mix_ratio))
        }
    };

    if verbosity.normal_verbosity() {
        println!("Built the {sym:?} graph: {:.2?}", start.elapsed());
    }
    Ok(graph)
}

/// Flatten a `Vec<Vec<usize>>` kNN into the row-major `u32` layout that
/// [`build_knn_graph`] takes.
///
/// ### Params
///
/// * `knn` - Neighbour indices per node, all rows of length `k`
///
/// ### Returns
///
/// `(idx, k)`
pub fn flatten_knn(knn: &[Vec<usize>]) -> (Vec<u32>, usize) {
    let k = knn.first().map_or(0, |r| r.len());
    let mut idx = Vec::with_capacity(knn.len() * k);
    for row in knn {
        idx.extend(row.iter().map(|&j| j as u32));
    }
    (idx, k)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // 4 nodes, k = 2, row-major:
    //   0 -> [1, 2], 1 -> [0, 2], 2 -> [0, 1], 3 -> [0, 1]
    fn small_knn() -> (Vec<u32>, usize, usize) {
        (vec![1, 2, 0, 2, 0, 1, 0, 1], 2, 4)
    }

    fn weight(g: &SparseGraph<f32>, a: usize, b: usize) -> Option<f32> {
        let (nb, w) = g.get_neighbours_raw(a);
        nb.iter().position(|&x| x as usize == b).map(|p| w[p])
    }

    fn all_weights(g: &SparseGraph<f32>) -> Vec<f32> {
        (0..g.get_node_number())
            .flat_map(|i| g.get_neighbours_raw(i).1.to_vec())
            .collect()
    }

    fn assert_symmetric_sorted(g: &SparseGraph<f32>) {
        for u in 0..g.get_node_number() {
            let (nb, w) = g.get_neighbours_raw(u);
            assert!(nb.windows(2).all(|p| p[0] < p[1]), "row {u} not sorted");
            for (&v, &wv) in nb.iter().zip(w) {
                assert_ne!(v as usize, u, "self loop at {u}");
                assert_eq!(weight(g, v as usize, u), Some(wv));
            }
        }
    }

    /// Seeded random kNN with distinct neighbours, self excluded, and
    /// ascending distances.
    fn random_knn(n: usize, k: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut idx = Vec::with_capacity(n * k);
        let mut dist = Vec::with_capacity(n * k);
        for i in 0..n {
            let mut row: Vec<u32> = Vec::with_capacity(k);
            while row.len() < k {
                // local neighbourhoods so the sNN is not empty
                let off = (next() % 40) as i64 - 20;
                let j = (i as i64 + off).rem_euclid(n as i64) as u32;
                if j as usize != i && !row.contains(&j) {
                    row.push(j);
                }
            }
            let mut d: Vec<f32> = (0..k)
                .map(|_| (next() % 1000) as f32 / 100.0 + 0.01)
                .collect();
            d.sort_by(f32::total_cmp);
            idx.extend(row);
            dist.extend(d);
        }
        (idx, dist)
    }

    /// Mutual pairs weigh 2, one-sided pairs 1; binary flattens both to 1.
    #[test]
    fn test_knn_union_weights() {
        let (idx, k, n) = small_knn();
        let g = build_knn_graph(
            &idx,
            None,
            k,
            n,
            &KnnSymmetrisation::Knn { binary: false },
            0,
        )
        .unwrap();
        assert_symmetric_sorted(&g);
        assert_eq!(weight(&g, 0, 1), Some(2.0));
        assert_eq!(weight(&g, 0, 3), Some(1.0));
        assert_eq!(weight(&g, 2, 3), None);
        let gb = build_knn_graph(
            &idx,
            None,
            k,
            n,
            &KnnSymmetrisation::Knn { binary: true },
            0,
        )
        .unwrap();
        assert_eq!(weight(&gb, 0, 1), Some(1.0));
        assert_eq!(gb.get_node_degree(0), 3);
    }

    /// Hand-checked Jaccard weights: (0,1) share {0,1,2}, (0,3) share {0,1}.
    #[test]
    fn test_snn_jaccard_known_weights() {
        let (idx, k, n) = small_knn();
        let sym = KnnSymmetrisation::Snn {
            similarity: SnnSimilarity::Jaccard,
            prune: 0.0,
            rescue: true,
        };
        let g = build_knn_graph::<f32>(&idx, None, k, n, &sym, 0).unwrap();
        assert_symmetric_sorted(&g);
        assert_eq!((0..n).map(|i| g.get_node_degree(i)).sum::<usize>(), 12);
        assert_relative_eq!(weight(&g, 0, 1).unwrap(), 1.0);
        assert_relative_eq!(weight(&g, 0, 3).unwrap(), 0.5);
        // (2,3): {0,1,2} vs {0,1,3} -> 2 / (6 - 2)
        assert_relative_eq!(weight(&g, 2, 3).unwrap(), 0.5);
    }

    /// Hand-checked rank weights: (0,1) share 0 at ranks 0 + 1, so
    /// `(2 - 1/2) / 2`.
    #[test]
    fn test_snn_rank_known_weights() {
        let (idx, k, n) = small_knn();
        let sym = KnnSymmetrisation::Snn {
            similarity: SnnSimilarity::Rank,
            prune: 0.0,
            rescue: false,
        };
        let g = build_knn_graph::<f32>(&idx, None, k, n, &sym, 0).unwrap();
        assert_relative_eq!(weight(&g, 0, 1).unwrap(), 0.75);
        // (2,3): best shared is 0 at ranks 1 + 1 -> (2 - 1) / 2
        assert_relative_eq!(weight(&g, 2, 3).unwrap(), 0.5);
        assert!(all_weights(&g).iter().all(|&w| w > 0.0 && w <= 1.0));
    }

    /// Pruning at 0.6 would isolate node 3; the rescue gives it back its
    /// strongest kNN edge, which ties at 0.5 and goes to the closer neighbour.
    #[test]
    fn test_snn_prune_and_rescue() {
        let (idx, k, n) = small_knn();
        let mk = |rescue| KnnSymmetrisation::Snn {
            similarity: SnnSimilarity::Jaccard,
            prune: 0.6,
            rescue,
        };
        let g = build_knn_graph::<f32>(&idx, None, k, n, &mk(true), 0).unwrap();
        assert_symmetric_sorted(&g);
        assert_eq!(weight(&g, 0, 3), Some(0.5));
        assert_eq!(g.get_node_degree(3), 1);
        let g0 = build_knn_graph::<f32>(&idx, None, k, n, &mk(false), 0).unwrap();
        assert_eq!(g0.get_node_degree(3), 0);
    }

    /// The sNN weights equal the pairwise definition on a random kNN.
    #[test]
    fn test_snn_matches_pairwise() {
        let (n, k) = (300, 8);
        let (idx, _) = random_knn(n, k, 7);
        for similarity in [SnnSimilarity::Jaccard, SnnSimilarity::Rank] {
            let sym = KnnSymmetrisation::Snn {
                similarity,
                prune: 0.0,
                rescue: false,
            };
            let g = build_knn_graph::<f32>(&idx, None, k, n, &sym, 0).unwrap();
            assert_symmetric_sorted(&g);
            for u in 0..n {
                let (nb, w) = g.get_neighbours_raw(u);
                for (&v, &wv) in nb.iter().zip(w) {
                    assert_eq!(
                        wv,
                        pair_snn_weight::<f32>(&idx, k, u, v as usize, similarity)
                    );
                }
            }
        }
    }

    /// Hand-checked UMAP memberships on two nodes pointing at each other:
    /// sigma is irrelevant at k = 1 because the only neighbour sits at rho.
    #[test]
    fn test_umap_two_nodes() {
        let idx = vec![1u32, 0];
        let dist = vec![0.5f32, 0.5];
        let sym = KnnSymmetrisation::Umap {
            local_connectivity: 1.0,
            mix_ratio: 1.0,
        };
        let g = build_knn_graph(&idx, Some(&dist), 1, 2, &sym, 0).unwrap();
        assert_eq!(weight(&g, 0, 1), Some(1.0));
    }

    /// Every node keeps its nearest neighbour at weight 1, so nothing is
    /// isolated, and the union is symmetric and in (0, 1].
    #[test]
    fn test_umap_no_singletons() {
        let (n, k) = (500, 10);
        let (idx, dist) = random_knn(n, k, 11);
        let sym = KnnSymmetrisation::Umap {
            local_connectivity: 1.0,
            mix_ratio: 1.0,
        };
        let g = build_knn_graph(&idx, Some(&dist), k, n, &sym, 0).unwrap();
        assert_symmetric_sorted(&g);
        for i in 0..n {
            assert!(g.get_node_degree(i) >= 1);
            // a + b - ab rounds below 1 in f32, as in umap-learn
            assert_relative_eq!(
                weight(&g, i, idx[i * k] as usize).unwrap(),
                1.0,
                epsilon = 1e-6
            );
        }
        assert!(all_weights(&g).iter().all(|&w| w > 0.0 && w <= 1.0));
    }

    /// The fuzzy union of a one-sided edge keeps its membership; the
    /// intersection (`mix_ratio = 0`) drops it.
    #[test]
    fn test_umap_mix_ratio() {
        let (idx, k, n) = small_knn();
        let dist = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let mk = |mix_ratio| KnnSymmetrisation::Umap {
            local_connectivity: 1.0,
            mix_ratio,
        };
        let gu = build_knn_graph(&idx, Some(&dist), k, n, &mk(1.0), 0).unwrap();
        let gi = build_knn_graph(&idx, Some(&dist), k, n, &mk(0.0), 0).unwrap();
        // 3 -> 0 is one-sided
        assert_eq!(weight(&gu, 0, 3), Some(1.0));
        assert_eq!(weight(&gi, 0, 3), None);
        assert!(gi.get_node_degree(0) < gu.get_node_degree(0));
    }

    /// Umap without distances, a short index and an out-of-range index are
    /// errors, not panics.
    #[test]
    fn test_build_knn_graph_errors() {
        let (idx, k, n) = small_knn();
        let umap = KnnSymmetrisation::Umap {
            local_connectivity: 1.0,
            mix_ratio: 1.0,
        };
        assert!(matches!(
            build_knn_graph::<f32>(&idx, None, k, n, &umap, 0),
            Err(MethodsErrors::KnnGraphNeedsDistances)
        ));
        assert!(matches!(
            build_knn_graph::<f32>(&idx[..7], None, k, n, &KnnSymmetrisation::default(), 0),
            Err(MethodsErrors::GraphLengthMismatch { .. })
        ));
        let mut bad = idx.clone();
        bad[0] = 9;
        assert!(matches!(
            build_knn_graph::<f32>(&bad, None, k, n, &KnnSymmetrisation::default(), 0),
            Err(MethodsErrors::GraphIndexOutOfRange { .. })
        ));
    }

    /// The parser is case-insensitive and refuses names it does not know.
    #[test]
    fn test_parse_snn_similarity() {
        assert_eq!(
            parse_snn_similarity("JACCARD"),
            Some(SnnSimilarity::Jaccard)
        );
        assert_eq!(parse_snn_similarity("rank"), Some(SnnSimilarity::Rank));
        assert_eq!(parse_snn_similarity("nope"), None);
    }
}
