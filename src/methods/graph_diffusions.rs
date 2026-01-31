use rayon::prelude::*;
use std::sync::Arc;

use crate::assert_same_len;
use crate::graph::graph_structures::graph_from_strings;
use crate::graph::page_rank::*;
use crate::prelude::BixverseFloat;

///////////
// Enums //
///////////

/// Enum for the TiedSumType types
#[derive(Clone, Debug, Default)]
pub enum TiedSumType {
    /// Minimum between the two diffusion vectors
    #[default]
    Min,
    /// Maximum between the two diffusion vectors
    Max,
    /// Average between the two diffusion vectors
    Avg,
}

/// Parsing the tied summarisation types
///
/// ### Params
///
/// * `s` - The string that defines the tied summarisation type
///
/// ### Results
///
/// The `TiedSumType` defining the tied summarisation type
pub fn parse_tied_sum(s: &str) -> Option<TiedSumType> {
    match s.to_lowercase().as_str() {
        "max" => Some(TiedSumType::Max),
        "min" => Some(TiedSumType::Min),
        "mean" => Some(TiedSumType::Avg),
        _ => None,
    }
}

////////////////////
// Main functions //
////////////////////

/// Calculates the tied diffusion between two graphs
///
/// Function is designed to handle permutations of the personalisation vectors
/// and leverages parallel processing for efficiency.
///
/// ### Params
///
/// * `node_names` - The names of the nodes in the graph
/// * `from` - The source nodes of the edges in the graph
/// * `to` - The destination nodes of the edges in the graph
/// * `weights` - The weights of the edges in the graph
/// * `summarisation_fun` - The function used to summarise the personalisation
///   vectors (i.e., the tied diffusion)
/// * `personalise_vecs_1` - The personalisation vectors for the first graph
/// * `personalise_vecs_2` - The personalisation vectors for the second graph
/// * `undirected` - Whether the graph is undirected
///
/// ### Returns
///
/// * `Vec<Vec<T>>` - The tied diffusion between the two graphs for all of the
///   supplied personalisation vectors
#[allow(clippy::too_many_arguments)]
pub fn tied_diffusion_parallel<T>(
    node_names: Vec<String>,
    from: Vec<String>,
    to: Vec<String>,
    weights: Option<&[T]>,
    summarisation_fun: String,
    personalise_vecs_1: &[Vec<T>],
    personalise_vecs_2: &[Vec<T>],
    undirected: bool,
) -> Vec<Vec<T>>
where
    T: BixverseFloat + std::iter::Sum,
{
    assert_same_len!(node_names, from, to, personalise_vecs_1, personalise_vecs_2);

    let summarisation_type: TiedSumType = parse_tied_sum(&summarisation_fun).unwrap();
    let graph_1 = graph_from_strings(&node_names, &from, &to, weights, undirected);

    // For tied diffusion in the directed case, the directionality is reversed
    let graph_2 = if !undirected {
        graph_from_strings(&node_names, &to, &from, weights, undirected)
    } else {
        graph_from_strings(&node_names, &from, &to, weights, undirected)
    };

    let dampening_factor = T::from_f64(0.85).unwrap();
    let tolerance = T::from_f64(1e-7).unwrap();
    let half = T::from_f64(0.5).unwrap();

    // Pre-process graph once
    let pagerank_graph_1 = Arc::new(PageRankGraph::from_petgraph(graph_1));
    let pagerank_graph_2 = Arc::new(PageRankGraph::from_petgraph(graph_2));

    let tied_res: Vec<Vec<T>> = personalise_vecs_1
        .into_par_iter()
        .zip(personalise_vecs_2.into_par_iter())
        .map_init(
            || (PageRankWorkingMemory::new(), PageRankWorkingMemory::new()),
            |(working_mem1, working_mem2), (diff1, diff2)| {
                let pr1 = personalised_page_rank_optimised(
                    &pagerank_graph_1,
                    dampening_factor,
                    diff1,
                    1000,
                    tolerance,
                    working_mem1,
                );
                let pr2 = personalised_page_rank_optimised(
                    &pagerank_graph_2,
                    dampening_factor,
                    diff2,
                    1000,
                    tolerance,
                    working_mem2,
                );

                pr1.iter()
                    .zip(pr2.iter())
                    .map(|(&v1, &v2)| match summarisation_type {
                        TiedSumType::Max => v1.max(v2),
                        TiedSumType::Min => v1.min(v2),
                        TiedSumType::Avg => (v1 + v2) * half, // multiplication is faster
                    })
                    .collect()
            },
        )
        .collect();

    tied_res
}
