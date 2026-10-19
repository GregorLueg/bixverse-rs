//! Methods to calculate ontological similarities (Wang or semantic similarity
//! implemented for now)

use faer::Mat;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::sync::RwLock;

use crate::graph::graph_structures::edges_to_csr;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Tile edge for the transpose pass that mirrors the Wang similarity triangle
const WANG_MIRROR_TILE: usize = 64;

///////////////////////////
// Semantic similarities //
///////////////////////////

/// Enum to define the different semantic similarity types
#[derive(Clone, Debug, Default)]
enum OntoSemSimType {
    #[default]
    Resnik,
    Lin,
    Combined,
}

/// Parse the semantic similarity type from a string
///
/// ### Params
///
/// * `sim_type` - The string to parse
///
/// ### Returns
///
/// The parsed semantic similarity type, or None if the string is invalid
fn parse_onto_similarity_type(sim_type: &str) -> Option<OntoSemSimType> {
    match sim_type.to_lowercase().as_str() {
        "resnik" => Some(OntoSemSimType::Resnik),
        "lin" => Some(OntoSemSimType::Lin),
        "combined" => Some(OntoSemSimType::Combined),
        _ => None,
    }
}

/// Structure to store the Ontology similarity results
#[derive(Clone, Debug)]
pub struct OntoSimRes<'a, T> {
    /// Name of term 1.
    pub t1: &'a str,
    /// Name of term 2.
    pub t2: &'a str,
    /// The calculated semantic or Wang similarity
    pub sim: T,
}

/// Terms interned to dense ids, with each term's ancestors that carry
/// information content as an id-sorted list
///
/// ### Fields
///
/// * `ids` - Term name to id
/// * `ancestors` - Per term id, sorted `(ancestor id, information content)`
/// * `term_ic` - Per term id, information content (1 when missing)
struct InternedOntology<'a, T> {
    ids: FxHashMap<&'a str, u32>,
    ancestors: Vec<Vec<(u32, T)>>,
    term_ic: Vec<T>,
}

impl<'a, T: BixverseFloat> InternedOntology<'a, T> {
    /// Intern the query terms and their ancestors
    ///
    /// ### Params
    ///
    /// * `terms_split` - The query terms, as given to `calculate_onto_sim`
    /// * `ancestor_map` - HashMap with the ancestors of the terms.
    /// * `info_content_map` - BTreeMap with the information content.
    ///
    /// ### Returns
    ///
    /// The interned structure.
    fn new(
        terms_split: &'a [(String, &[String])],
        ancestor_map: &'a FxHashMap<String, FxHashSet<String>>,
        info_content_map: &'a BTreeMap<String, T>,
    ) -> Self {
        let mut ids: FxHashMap<&'a str, u32> = FxHashMap::default();
        let mut terms: Vec<&'a str> = Vec::new();
        for (t1, others) in terms_split {
            for term in std::iter::once(t1).chain(others.iter()) {
                ids.entry(term.as_str()).or_insert_with(|| {
                    terms.push(term.as_str());
                    (terms.len() - 1) as u32
                });
            }
        }

        let n_terms = terms.len();
        let mut ancestors: Vec<Vec<(u32, T)>> = Vec::with_capacity(n_terms);
        let mut term_ic: Vec<T> = Vec::with_capacity(n_terms);
        let mut next_id = n_terms as u32;

        for &term in &terms {
            term_ic.push(info_content_map.get(term).copied().unwrap_or(T::one()));

            let mut list: Vec<(u32, T)> = Vec::new();
            if let Some(set) = ancestor_map.get(term) {
                for ancestor in set {
                    if let Some(&ic) = info_content_map.get(ancestor) {
                        let id = *ids.entry(ancestor.as_str()).or_insert_with(|| {
                            next_id += 1;
                            next_id - 1
                        });
                        list.push((id, ic));
                    }
                }
            }
            list.sort_unstable_by_key(|&(id, _)| id);
            ancestors.push(list);
        }

        Self {
            ids,
            ancestors,
            term_ic,
        }
    }

    /// Information content of the most informative common ancestor
    ///
    /// ### Params
    ///
    /// * `a` - Id of term 1.
    /// * `b` - Id of term 2.
    ///
    /// ### Returns
    ///
    /// The maximum information content over the common ancestors, zero if none.
    #[inline]
    fn mica(&self, a: usize, b: usize) -> T {
        let (xs, ys) = (&self.ancestors[a], &self.ancestors[b]);
        let (mut i, mut j) = (0, 0);
        let mut best = T::zero();
        while i < xs.len() && j < ys.len() {
            match xs[i].0.cmp(&ys[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    best = best.max(xs[i].1);
                    i += 1;
                    j += 1;
                }
            }
        }
        best
    }
}

/// Calculate the semantic similarity in an efficient manner for a set of terms
///
/// ### Params
///
/// * `terms_split` - A vector of tuples with the first element being the term 1
///   and the second element being the terms against which to calculate
///   the semantic similarity.
/// * `sim_type` - Which type of semantic similarity to calculate.
/// * `ancestor_map` - HashMap with the ancestors of the terms.
/// * `ic_map` - HashMap with the information content for the terms.
///
/// ### Returns
///
/// A vector of `OntoSimRes` results.
pub fn calculate_onto_sim<'a, T>(
    terms_split: &'a Vec<(String, &[String])>,
    sim_type: &str,
    ancestors_map: FxHashMap<String, FxHashSet<String>>,
    ic_map: BTreeMap<String, T>,
) -> Vec<OntoSimRes<'a, T>>
where
    T: BixverseFloat,
{
    let max_ic = *ic_map
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let sim_type = parse_onto_similarity_type(sim_type).unwrap_or_default();

    let interned = InternedOntology::new(terms_split, &ancestors_map, &ic_map);

    let half = T::from_f32(0.5).unwrap();
    let two = T::from_f32(2.0).unwrap();

    let onto_sim: Vec<Vec<OntoSimRes<'_, T>>> = terms_split
        .par_iter()
        .map(|(t1, others)| {
            let id1 = interned.ids[t1.as_str()] as usize;
            let ic1 = interned.term_ic[id1];

            others
                .iter()
                .map(|t2| {
                    let id2 = interned.ids[t2.as_str()] as usize;
                    let mica = interned.mica(id1, id2);
                    let ic2 = interned.term_ic[id2];

                    let sim = match sim_type {
                        OntoSemSimType::Resnik => mica / max_ic,
                        OntoSemSimType::Lin => two * mica / (ic1 + ic2),
                        OntoSemSimType::Combined => {
                            let lin_sim = two * mica / (ic1 + ic2);
                            let resnik_sim = mica / max_ic;
                            (lin_sim + resnik_sim) * half
                        }
                    };

                    OntoSimRes { t1, t2, sim }
                })
                .collect()
        })
        .collect();

    flatten_vector(onto_sim)
}

///////////////////////
// DAG-based methods //
///////////////////////

/// Type alias for SValue Cache
///
/// This one needs the RwLock to be able to do the parallelisation on top. This
/// locks it to maximum one writer and multiple readers at the same time.
pub type SValueCache<T> = RwLock<FxHashMap<usize, FxHashMap<usize, T>>>;

/// Structure for calculating the Wang similarity on a given Ontology
pub struct WangSimOntology<T> {
    /// HashMap between term to node index.
    term_to_idx: FxHashMap<String, usize>,
    /// The term order as a string.
    idx_to_term: Vec<String>,
    /// Offsets into `child_indices` per parent, length `n_terms + 1`
    child_indptr: Vec<usize>,
    /// Child node per parent -> child edge. Duplicate edges are kept.
    child_indices: Vec<u32>,
    /// Weight per entry of `child_indices`
    child_weights: Vec<T>,
    /// A vector containing the ancestors as HashSets.
    ancestors: Vec<FxHashSet<usize>>,
    /// Topological order, children before parents.
    topo_order: Vec<usize>,
    /// The `SValueCache` caching all of the S values for each node.
    s_values_cache: SValueCache<T>,
}

impl<T> WangSimOntology<T>
where
    T: BixverseFloat + std::iter::Sum,
{
    /// Create a new ontology object from parents, child and weights between them
    ///
    /// ### Params
    ///
    /// * `parents` - Slice containing the names of the parents
    /// * `children` - Slice containing the names of the children
    /// * `w` - Slice of the weights between the parents and children.
    ///
    /// ### Returns
    ///
    /// The initialised `WangSimOntology`.
    ///
    /// ### Panics
    ///
    /// If the parent -> child graph contains a cycle.
    pub fn new(parents: &[String], children: &[String], w: &[T]) -> Self {
        assert_same_len!(parents, children, w);

        let mut term_to_idx = FxHashMap::default();
        let mut idx_to_term = Vec::new();

        // More efficient collection of unique terms
        let mut all_terms = FxHashSet::with_capacity_and_hasher(
            (parents.len() + children.len()) * 2,
            FxBuildHasher,
        );
        all_terms.extend(parents.iter().cloned());
        all_terms.extend(children.iter().cloned());

        // Pre-allocate vectors with known capacity
        idx_to_term.reserve(all_terms.len());
        term_to_idx.reserve(all_terms.len());

        for term in all_terms {
            term_to_idx.insert(term.clone(), idx_to_term.len());
            idx_to_term.push(term);
        }

        let n = idx_to_term.len();
        let parent_idx: Vec<u32> = parents.iter().map(|t| term_to_idx[t] as u32).collect();
        let child_idx: Vec<u32> = children.iter().map(|t| term_to_idx[t] as u32).collect();

        let (child_indptr, child_indices, child_weights) =
            edges_to_csr(n, &parent_idx, &child_idx, w);
        let (parent_indptr, parent_indices, _) = edges_to_csr(n, &child_idx, &parent_idx, w);

        let topo_order = Self::compute_topological_order(&child_indptr, &child_indices);
        let ancestors = Self::get_ancestors(&parent_indptr, &parent_indices, &topo_order);

        WangSimOntology {
            term_to_idx,
            idx_to_term,
            child_indptr,
            child_indices,
            child_weights,
            ancestors,
            topo_order,
            s_values_cache: RwLock::new(FxHashMap::default()),
        }
    }

    /// Calculate the similarity matrix with optimizations
    ///
    /// ### Returns
    ///
    /// A tuple with the full Wang similarity matrix as first element and the
    /// column/row names as the second element.
    pub fn calc_sim_matrix(&self) -> (Mat<T>, Vec<String>) {
        let n = self.idx_to_term.len();
        let mut matrix: Mat<T> = Mat::zeros(n, n);

        // Per term: S-values over every ancestor (zero where absent), sorted by
        // node index, plus their total.
        let per_term: Vec<(Vec<(u32, T)>, T)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let s_values = self.get_or_compute_s_values(i);
                let total: T = s_values.values().copied().sum();
                let mut dense: Vec<(u32, T)> = self.ancestors[i]
                    .iter()
                    .map(|node| {
                        (
                            *node as u32,
                            s_values.get(node).copied().unwrap_or_else(T::zero),
                        )
                    })
                    .collect();
                dense.sort_unstable_by_key(|&(id, _)| id);
                (dense, total)
            })
            .collect();

        // Upper triangle straight into each column, then a tiled mirror.
        matrix
            .par_col_iter_mut()
            .enumerate()
            .for_each(|(j, mut col)| {
                for i in 0..j {
                    col[i] = Self::pair_similarity(
                        &per_term[i].0,
                        &per_term[j].0,
                        per_term[i].1,
                        per_term[j].1,
                    );
                }
                col[j] = T::one();
            });

        for jj in (0..n).step_by(WANG_MIRROR_TILE) {
            for ii in (jj..n).step_by(WANG_MIRROR_TILE) {
                for j in jj..(jj + WANG_MIRROR_TILE).min(n) {
                    for i in ii.max(j + 1)..(ii + WANG_MIRROR_TILE).min(n) {
                        matrix[(i, j)] = matrix[(j, i)];
                    }
                }
            }
        }

        (matrix, self.idx_to_term.clone())
    }

    /// Clear the S-values cache (useful for memory management)
    pub fn clear_cache(&self) {
        let mut cache = self.s_values_cache.write().unwrap();
        cache.clear();
    }

    /// Get or compute S-values with caching
    ///
    /// ### Params
    ///
    /// * `term_idx` The node index for which to calculate the S value
    ///
    /// ### Returns
    ///
    /// Returns a HashMap of the node indices with the given S values for all ancestors
    /// of the specified term_idx.
    fn get_or_compute_s_values(&self, term_idx: usize) -> FxHashMap<usize, T> {
        // Try to read from cache first
        {
            let cache = self.s_values_cache.read().unwrap();
            if let Some(cached) = cache.get(&term_idx) {
                return cached.clone();
            }
        }

        // Compute and store if not in cache
        let s_values = self.calculate_s_values(term_idx);

        {
            let mut cache = self.s_values_cache.write().unwrap();
            cache.insert(term_idx, s_values.clone());
        }

        s_values
    }

    /// Calculate similarity from pre-computed, id-sorted S-values
    ///
    /// ### Params
    ///
    /// * `s_val_1` - Ancestors of term 1 with their S-values (zero if absent),
    ///   sorted by node index.
    /// * `s_val_2` - Same for term 2.
    /// * `sv1` - Total of the S-values of term 1.
    /// * `sv2` - Total of the S-values of term 2.
    ///
    /// ### Returns
    ///
    /// The Wang similarity between the two terms.
    #[inline]
    fn pair_similarity(s_val_1: &[(u32, T)], s_val_2: &[(u32, T)], sv1: T, sv2: T) -> T {
        let zero = T::zero();
        let (mut i, mut j) = (0, 0);
        let mut numerator = zero;
        let mut any_common = false;

        while i < s_val_1.len() && j < s_val_2.len() {
            match s_val_1[i].0.cmp(&s_val_2[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    numerator += s_val_1[i].1 + s_val_2[j].1;
                    any_common = true;
                    i += 1;
                    j += 1;
                }
            }
        }

        if !any_common {
            return zero;
        }

        let denominator = sv1 + sv2;

        if denominator > zero {
            numerator / denominator
        } else {
            zero
        }
    }

    /// Get the ancestor terms of everything in the ontology
    ///
    /// ### Params
    ///
    /// * `parent_indptr` - Offsets into `parent_indices` per child
    /// * `parent_indices` - Parent node per child -> parent edge
    /// * `topo_order` - Topological order, children before parents
    ///
    /// ### Returns
    ///
    /// A vector of the HashSets with the ancestor node indices, self included.
    fn get_ancestors(
        parent_indptr: &[usize],
        parent_indices: &[u32],
        topo_order: &[usize],
    ) -> Vec<FxHashSet<usize>> {
        let mut ancestors = vec![FxHashSet::default(); topo_order.len()];

        // Parents before children, so every parent set is final when read
        for &node_idx in topo_order.iter().rev() {
            let mut node_ancestors = FxHashSet::default();
            node_ancestors.insert(node_idx);
            for &parent_idx in &parent_indices[parent_indptr[node_idx]..parent_indptr[node_idx + 1]]
            {
                node_ancestors.extend(&ancestors[parent_idx as usize]);
            }
            ancestors[node_idx] = node_ancestors;
        }

        ancestors
    }

    /// Compute the topological order via Kahn's algorithm
    ///
    /// ### Params
    ///
    /// * `child_indptr` - Offsets into `child_indices` per parent
    /// * `child_indices` - Child node per parent -> child edge
    ///
    /// ### Returns
    ///
    /// The node indices in topological order, children before parents.
    ///
    /// ### Panics
    ///
    /// If the graph contains a cycle.
    fn compute_topological_order(child_indptr: &[usize], child_indices: &[u32]) -> Vec<usize> {
        let n = child_indptr.len() - 1;
        let mut in_degree = vec![0usize; n];
        for &c in child_indices {
            in_degree[c as usize] += 1;
        }

        let mut order: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut head = 0;
        while head < order.len() {
            let node = order[head];
            head += 1;
            for &c in &child_indices[child_indptr[node]..child_indptr[node + 1]] {
                in_degree[c as usize] -= 1;
                if in_degree[c as usize] == 0 {
                    order.push(c as usize);
                }
            }
        }

        if order.len() < n {
            panic!("Ontology contains cycles");
        }

        order.reverse();
        order
    }

    /// Calculate the S-values for a specific term's DAG
    ///
    /// ### Params
    ///
    /// * `term_idx` - The node index for which to calculate the S values
    ///
    /// ### Returns
    ///
    /// A HashMap of node indices with corresponding S values
    fn calculate_s_values(&self, term_idx: usize) -> FxHashMap<usize, T> {
        let dag_nodes = &self.ancestors[term_idx];
        let mut s_values = FxHashMap::with_capacity_and_hasher(dag_nodes.len(), FxBuildHasher);

        s_values.insert(term_idx, T::one());

        // Process in topological order (children before parents)
        for &node_idx in &self.topo_order {
            if !dag_nodes.contains(&node_idx) || node_idx == term_idx {
                continue;
            }

            let mut max_contribution: T = T::zero();

            let range = self.child_indptr[node_idx]..self.child_indptr[node_idx + 1];
            for (&child_idx, &edge_weight) in self.child_indices[range.clone()]
                .iter()
                .zip(&self.child_weights[range])
            {
                let child_idx = child_idx as usize;

                if dag_nodes.contains(&child_idx)
                    && let Some(&child_s_value) = s_values.get(&child_idx)
                {
                    max_contribution = max_contribution.max(edge_weight * child_s_value);
                }
            }

            if max_contribution > T::zero() {
                s_values.insert(node_idx, max_contribution);
            }
        }

        s_values
    }

    /// Calculate Wang similarity between two terms with caching
    ///
    /// ### Params
    ///
    /// * `term1` - Name of term1
    /// * `term2` - Name of term2
    ///
    /// ### Returns
    ///
    /// The optional Wang similarity between the two terms.
    pub fn wang_sim(&self, term1: &str, term2: &str) -> Option<T> {
        let term_idx_1 = *self.term_to_idx.get(term1)?;
        let term_idx_2 = *self.term_to_idx.get(term2)?;

        if term_idx_1 == term_idx_2 {
            return Some(T::one());
        }

        self.wang_sim_by_idx(term_idx_1, term_idx_2)
    }

    /// Calculate Wang similarity between two term indices (internal method)
    ///
    /// ### Params
    ///
    /// * `term_idx_1` - Node index of term 1
    /// * `term_idx_2` - Node index of term 2
    ///
    /// ### Returns
    ///
    /// The Wang similarity between the two terms.
    fn wang_sim_by_idx(&self, term_idx_1: usize, term_idx_2: usize) -> Option<T> {
        let s_val_1 = self.get_or_compute_s_values(term_idx_1);
        let s_val_2 = self.get_or_compute_s_values(term_idx_2);

        let dag1_nodes = &self.ancestors[term_idx_1];
        let dag2_nodes = &self.ancestors[term_idx_2];

        // Find intersection more efficiently
        let (smaller, larger) = if dag1_nodes.len() < dag2_nodes.len() {
            (dag1_nodes, dag2_nodes)
        } else {
            (dag2_nodes, dag1_nodes)
        };

        let common_nodes: Vec<usize> = smaller
            .iter()
            .filter(|node| larger.contains(node))
            .cloned()
            .collect();

        // No shared ancestor means no shared semantics, so the similarity is 0.
        // This used to return 1.0, which disagreed with
        // [Self::pair_similarity] (the path `calc_sim_matrix`
        // takes) and reported maximum similarity for terms in disconnected parts
        // of the DAG, e.g. any GO BP term against any GO MF term.
        if common_nodes.is_empty() {
            return Some(T::zero());
        }

        let sv1: T = s_val_1.values().copied().sum();
        let sv2: T = s_val_2.values().copied().sum();
        let zero = T::zero();

        let numerator: T = common_nodes
            .iter()
            .map(|&node_idx| {
                *s_val_1.get(&node_idx).unwrap_or(&zero) + *s_val_2.get(&node_idx).unwrap_or(&zero)
            })
            .sum();

        let denominator = sv1 + sv2;

        if denominator > zero {
            Some(numerator / denominator)
        } else {
            Some(T::zero())
        }
    }
}

////////////
// Others //
////////////

/// Filter similarities based on threshold
///
/// Helper function to generate a Vector of `OntoSimRes` results based on the
/// row-major similarities (as a vector), a threshold and the col/row names
/// of the similarity matrix
///
/// ### Params
///
/// * `sim_vals` - The upper triangle values of the similarity matrix stored in
///   row major format, excluding the diagonal.
/// * `names` - The column and row names of the similarity matrix.
/// * `threshold` - Filtering threshold
///
/// ### Returns
///
/// A vector of `OntoSimRes` that pass the threshold.
pub fn filter_sims_critval<'a, T>(
    sim_vals: &[T],
    names: &'a [String],
    threshold: T,
) -> Vec<OntoSimRes<'a, T>>
where
    T: BixverseFloat,
{
    let n = names.len();
    let mut results = Vec::new();
    let mut idx = 0;

    for i in 0..n {
        for j in i..n {
            if i != j {
                if idx < sim_vals.len() {
                    let sim = sim_vals[idx];
                    if sim >= threshold {
                        results.push(OntoSimRes {
                            t1: &names[i],
                            t2: &names[j],
                            sim,
                        })
                    }
                }
                idx += 1;
            }
        }
    }

    results
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The chain `A -> B -> C` at edge weight 0.8. Ancestor sets include self,
    /// so by hand:
    ///   S_C = {C: 1, B: 0.8, A: 0.64}, SV(C) = 2.44
    ///   S_B = {B: 1, A: 0.8},          SV(B) = 1.80
    ///   S_A = {A: 1},                  SV(A) = 1.00
    fn chain() -> WangSimOntology<f64> {
        WangSimOntology::new(
            &["A".to_string(), "B".to_string()],
            &["B".to_string(), "C".to_string()],
            &[0.8, 0.8],
        )
    }

    /// Every pair of the chain, derived from the S-values above.
    #[test]
    fn wang_sim_on_a_chain_matches_the_hand_derivation() {
        let onto = chain();

        // (0.8 + 1.0) + (0.64 + 0.8) = 3.24 over 2.44 + 1.80
        assert_relative_eq!(
            onto.wang_sim("B", "C").unwrap(),
            3.24 / 4.24,
            epsilon = 1e-12
        );
        // (0.64 + 1.0) over 2.44 + 1.00
        assert_relative_eq!(
            onto.wang_sim("A", "C").unwrap(),
            1.64 / 3.44,
            epsilon = 1e-12
        );
        // (0.8 + 1.0) over 1.80 + 1.00
        assert_relative_eq!(onto.wang_sim("A", "B").unwrap(), 1.8 / 2.8, epsilon = 1e-12);
    }

    /// A term against itself is 1, and an unknown term is `None`.
    #[test]
    fn wang_sim_handles_self_and_unknown_terms() {
        let onto = chain();

        assert_relative_eq!(onto.wang_sim("A", "A").unwrap(), 1.0, epsilon = 1e-12);
        assert!(onto.wang_sim("A", "NOPE").is_none());
    }

    /// Regression: terms in disconnected parts of the DAG share no ancestor, so
    /// the similarity is 0. `wang_sim` used to return 1.0 here while
    /// `calc_sim_matrix` returned 0.0, i.e. maximum similarity for maximally
    /// dissimilar terms. Asserting the two paths agree is the tightest way to
    /// state that.
    #[test]
    fn wang_sim_of_disconnected_terms_is_zero_and_matches_the_matrix() {
        let onto = WangSimOntology::<f64>::new(
            &["X".to_string(), "Y".to_string()],
            &["X1".to_string(), "Y1".to_string()],
            &[0.8, 0.8],
        );

        let pair = onto.wang_sim("X1", "Y1").unwrap();
        assert_relative_eq!(pair, 0.0, epsilon = 1e-12);

        let (matrix, names) = onto.calc_sim_matrix();
        let i = names.iter().position(|n| n == "X1").unwrap();
        let j = names.iter().position(|n| n == "Y1").unwrap();
        assert_relative_eq!(pair, matrix[(i, j)], epsilon = 1e-12);
    }

    /// The matrix agrees with the pairwise accessor, is symmetric, and has a
    /// unit diagonal. Term order is hash-driven, so look everything up by name.
    #[test]
    fn calc_sim_matrix_is_symmetric_and_agrees_with_wang_sim() {
        let onto = chain();
        let (matrix, names) = onto.calc_sim_matrix();

        assert_eq!(names.len(), 3);
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["A", "B", "C"]);

        for (i, ni) in names.iter().enumerate() {
            assert_relative_eq!(matrix[(i, i)], 1.0, epsilon = 1e-12);
            for (j, nj) in names.iter().enumerate() {
                assert_relative_eq!(matrix[(i, j)], matrix[(j, i)], epsilon = 1e-12);
                if i != j {
                    assert_relative_eq!(
                        matrix[(i, j)],
                        onto.wang_sim(ni, nj).unwrap(),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    /// A cycle has no topological order, so construction must refuse it.
    #[test]
    #[should_panic(expected = "cycle")]
    fn wang_sim_ontology_rejects_a_cyclic_graph() {
        WangSimOntology::<f64>::new(
            &["A".to_string(), "B".to_string()],
            &["B".to_string(), "A".to_string()],
            &[0.8, 0.8],
        );
    }
}
