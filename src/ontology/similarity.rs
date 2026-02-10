use faer::Mat;
use once_cell::sync::Lazy;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::prelude::*;

/////////////
// Globals //
/////////////

/// Static global for re-use
static EMPTY_ANCESTORS: Lazy<FxHashSet<String>> = Lazy::new(FxHashSet::default);

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
///
/// ### Fields
///
/// * `t1` - Name of term 1.
/// * `t2` - Name of term 2.
/// * `sim` - The calculated semantic or Wang similarity
#[derive(Clone, Debug)]
pub struct OntoSimRes<'a, T> {
    pub t1: &'a str,
    pub t2: &'a str,
    pub sim: T,
}

/// Get the information content of the MICA
///
/// ### Params
///
/// * `t1` - Name of term 1.
/// * `t2` - Name of term 2.
/// * `ancestor_map` - HashMap with the ancestors of the terms.
/// * `info_content_map` - HashMap with the information content for the terms.
///
/// ### Returns
///
/// The information content of the most informative common ancestors.
#[inline]
fn get_mica<T>(
    t1: &str,
    t2: &str,
    ancestor_map: &FxHashMap<String, FxHashSet<String>>,
    info_content_map: &BTreeMap<String, T>,
) -> T
where
    T: BixverseFloat,
{
    let ancestor_1 = ancestor_map.get(t1).unwrap_or(&EMPTY_ANCESTORS);
    let ancestor_2 = ancestor_map.get(t2).unwrap_or(&EMPTY_ANCESTORS);

    let (smaller, larger) = if ancestor_1.len() <= ancestor_2.len() {
        (ancestor_1, ancestor_2)
    } else {
        (ancestor_2, ancestor_1)
    };

    smaller
        .iter()
        .filter(|ancestor| larger.contains(*ancestor))
        .filter_map(|ancestor| info_content_map.get(ancestor))
        .fold(T::zero(), |max_ic, &ic| max_ic.max(ic))
}

/// Calculate semantic similarity between two terms
///
/// ### Params
///
/// * `t1` - Name of term 1.
/// * `t2` - Name of term 2.
/// * `sim_type` - `OntoSemSimType` defining the type of semantic similarity
///   to calculate.
/// * `max_ic` - The maximum information content observed to rescale the Resnik
///   similarity between 0 and 1.
/// * `ancestor_map` - HashMap with the ancestors of the terms.
/// * `info_content_map` - BTreeMap with the information content for the terms.
///
/// ### Returns
///
/// `OntoSimRes` result.
fn calculate_onto_similarity<'a, T>(
    t1: &'a str,
    t2: &'a str,
    sim_type: &OntoSemSimType,
    max_ic: T,
    ancestor_map: &FxHashMap<String, FxHashSet<String>>,
    info_content_map: &BTreeMap<String, T>,
) -> OntoSimRes<'a, T>
where
    T: BixverseFloat,
{
    let mica = get_mica(t1, t2, ancestor_map, info_content_map);

    let half = T::from_f32(0.5).unwrap();
    let two = T::from_f32(2.0).unwrap();
    let one = T::one();

    let sim = match sim_type {
        OntoSemSimType::Resnik => mica / max_ic,
        OntoSemSimType::Lin => {
            let t1_ic = info_content_map.get(t1).unwrap_or(&one);
            let t2_ic = info_content_map.get(t2).unwrap_or(&one);
            two * mica / (*t1_ic + *t2_ic)
        }
        OntoSemSimType::Combined => {
            let t1_ic = info_content_map.get(t1).unwrap_or(&one);
            let t2_ic = info_content_map.get(t2).unwrap_or(&one);
            let lin_sim = two * mica / (*t1_ic + *t2_ic);
            let resnik_sim = mica / max_ic;
            (lin_sim + resnik_sim) * half
        }
    };

    OntoSimRes { t1, t2, sim }
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
    let max_ic = ic_map
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let sim_type = parse_onto_similarity_type(sim_type).unwrap_or_default();

    let onto_sim: Vec<Vec<OntoSimRes<'_, T>>> = terms_split
        .par_iter()
        .map(|(t1, others)| {
            let mut sim_vec: Vec<OntoSimRes<'_, T>> = Vec::with_capacity(others.len());
            others.iter().for_each(|t2| {
                let sim_res =
                    calculate_onto_similarity(t1, t2, &sim_type, *max_ic, &ancestors_map, &ic_map);
                sim_vec.push(sim_res);
            });

            sim_vec
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
pub type SValueCache<T> = RwLock<FxHashMap<NodeIndex, FxHashMap<NodeIndex, T>>>;

/// Structure for calculating the Wang similarity on a given Ontology
///
/// ### Fields
///
/// * `term_to_idx` - HashMap between term to node index.
/// * `idx_to_term` - The term order as a string.
/// * `graph` - Directed Graph from parent to child.
/// * `ancestors` - A vector containing the ancestors as HashSets.
/// * `topo_order` - Calculated topological order.
/// * `s_values_cache` - The `SValueCache` caching all of the S values for each
///   node.
pub struct WangSimOntology<T> {
    term_to_idx: FxHashMap<String, NodeIndex>,
    idx_to_term: Vec<String>,
    graph: DiGraph<String, T>,
    ancestors: Vec<FxHashSet<NodeIndex>>,
    topo_order: Vec<NodeIndex>,
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
    pub fn new(parents: &[String], children: &[String], w: &[T]) -> Self {
        assert_same_len!(parents, children, w);

        let mut graph = DiGraph::new();
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

        // Add nodes to the graph
        for term in all_terms {
            let idx = graph.add_node(term.clone());
            term_to_idx.insert(term.clone(), idx);
            idx_to_term.push(term);
        }

        // Add edges to the graph
        for ((parent, child), w) in parents.iter().zip(children.iter()).zip(w.iter()) {
            let parent_idx = *term_to_idx.get(parent).unwrap();
            let child_idx = *term_to_idx.get(child).unwrap();
            graph.add_edge(parent_idx, child_idx, *w);
        }

        let topo_order = Self::compute_topological_order(&graph);
        let ancestors = Self::get_ancestors(&graph, &topo_order);

        WangSimOntology {
            term_to_idx,
            idx_to_term,
            graph,
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

        // Pre-compute all S-values once and store in Arc for safe sharing
        let all_s_values: Arc<Vec<FxHashMap<NodeIndex, T>>> = Arc::new(
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let term_idx = NodeIndex::new(i);
                    self.get_or_compute_s_values(term_idx)
                })
                .collect(),
        );

        // Process each row in parallel; mutex is a new one...
        let matrix_mutex = Mutex::new(&mut matrix);

        (0..n).into_par_iter().for_each(|i| {
            let mut row_values = Vec::with_capacity(n - i);

            // Calculate similarities for this row (only upper triangle)
            for j in i..n {
                let sim = if i == j {
                    T::one()
                } else {
                    let term_idx_1 = NodeIndex::new(i);
                    let term_idx_2 = NodeIndex::new(j);
                    self.calculate_similarity_from_s_values(
                        term_idx_1,
                        term_idx_2,
                        &all_s_values[i],
                        &all_s_values[j],
                    )
                };
                row_values.push((j, sim));
            }

            // Write the computed values to the matrix
            {
                let mut matrix_guard = matrix_mutex.lock().unwrap();
                for (j, sim) in row_values {
                    matrix_guard[(i, j)] = sim;
                    if i != j {
                        matrix_guard[(j, i)] = sim; // Symmetric matrix
                    }
                }
            }
        });

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
    /// Returns a HashMap of the NodeIndeces with the given S values for all ancestors
    /// of the specified term_idx.
    fn get_or_compute_s_values(&self, term_idx: NodeIndex) -> FxHashMap<NodeIndex, T> {
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

    /// Calculate similarity from pre-computed S-values
    ///
    /// ### Params
    ///
    /// * `term_idx_1` - NodeIndex of the first term
    /// * `term_idx_2` - NodeIndex of the second term
    /// * `s_val_1` - HashMap with the ancestors of term1 and the s-values.
    /// * `s_val_2` - HashMap with the ancestors of term2 and the s-values.
    ///
    /// ### Returns
    ///
    /// The Wang similarity between the two terms defined by the NodeIndex.
    fn calculate_similarity_from_s_values(
        &self,
        term_idx_1: NodeIndex,
        term_idx_2: NodeIndex,
        s_val_1: &FxHashMap<NodeIndex, T>,
        s_val_2: &FxHashMap<NodeIndex, T>,
    ) -> T {
        let dag1_nodes = &self.ancestors[term_idx_1.index()];
        let dag2_nodes = &self.ancestors[term_idx_2.index()];

        // Start with smaller to make it faster
        let (smaller, larger) = if dag1_nodes.len() < dag2_nodes.len() {
            (dag1_nodes, dag2_nodes)
        } else {
            (dag2_nodes, dag1_nodes)
        };

        let common_nodes: Vec<NodeIndex> = smaller
            .iter()
            .filter(|node| larger.contains(node))
            .cloned()
            .collect();

        if common_nodes.is_empty() {
            return T::zero();
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
            numerator / denominator
        } else {
            zero
        }
    }

    /// Get the ancestor terms of everything in the ontology
    ///
    /// ### Params
    ///
    /// * `graph` - The DirectedGraph representing the ontology
    /// * `topo_order` - The vector defining the topological order
    ///
    /// ### Returns
    ///
    /// A vector of the HashSets with the ancestor node indices.
    fn get_ancestors(
        graph: &DiGraph<String, T>,
        topo_order: &[NodeIndex],
    ) -> Vec<FxHashSet<NodeIndex>> {
        let mut ancestors = vec![FxHashSet::default(); graph.node_count()];

        // Process nodes in reverse topological order
        for &node_idx in topo_order.iter().rev() {
            let mut node_ancestors = FxHashSet::default();

            // Add self
            node_ancestors.insert(node_idx);
            for parent_idx in graph.neighbors_directed(node_idx, petgraph::Incoming) {
                node_ancestors.extend(&ancestors[parent_idx.index()]);
            }

            ancestors[node_idx.index()] = node_ancestors;
        }

        ancestors
    }

    /// Compute topological order
    ///
    /// ### Params
    ///
    /// * `graph` - The DirectedGraph representing the ontology
    ///
    /// ### Returns
    ///
    /// A vector of node indices based on the topological order of the directed
    /// graph.
    fn compute_topological_order(graph: &DiGraph<String, T>) -> Vec<NodeIndex> {
        petgraph::algo::toposort(graph, None)
            .unwrap_or_else(|_| panic!("Ontology contains cycles"))
            .into_iter()
            .rev()
            .collect()
    }

    /// Calculate the S-values for a specific term's DAG
    ///
    /// ### Params
    ///
    /// * `term_idx` - The NodeIndex for which to calculate the S values
    ///
    /// ### Returns
    ///
    /// A HashMap of NodeIndices with corresponding S values
    fn calculate_s_values(&self, term_idx: NodeIndex) -> FxHashMap<NodeIndex, T> {
        let dag_nodes = &self.ancestors[term_idx.index()];
        let mut s_values = FxHashMap::with_capacity_and_hasher(dag_nodes.len(), FxBuildHasher);

        s_values.insert(term_idx, T::one());

        // Process in topological order (children before parents)
        for &node_idx in &self.topo_order {
            if !dag_nodes.contains(&node_idx) || node_idx == term_idx {
                continue;
            }

            let mut max_contribution: T = T::zero();

            // Iterate through outgoing edges to get the specific weights
            for edge_ref in self.graph.edges_directed(node_idx, petgraph::Outgoing) {
                let child_idx = edge_ref.target();
                let edge_weight = *edge_ref.weight();

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
    /// * `term_idx_1` - NodeIndex of term 1
    /// * `term_idx_2` - NodeIndex of term 2
    ///
    /// ### Returns
    ///
    /// The Wang similarity between the two terms.
    fn wang_sim_by_idx(&self, term_idx_1: NodeIndex, term_idx_2: NodeIndex) -> Option<T> {
        let s_val_1 = self.get_or_compute_s_values(term_idx_1);
        let s_val_2 = self.get_or_compute_s_values(term_idx_2);

        let dag1_nodes = &self.ancestors[term_idx_1.index()];
        let dag2_nodes = &self.ancestors[term_idx_2.index()];

        // Find intersection more efficiently
        let (smaller, larger) = if dag1_nodes.len() < dag2_nodes.len() {
            (dag1_nodes, dag2_nodes)
        } else {
            (dag2_nodes, dag1_nodes)
        };

        let common_nodes: Vec<NodeIndex> = smaller
            .iter()
            .filter(|node| larger.contains(node))
            .cloned()
            .collect();

        if common_nodes.is_empty() {
            return Some(T::one());
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
