use rayon::prelude::*;

use crate::prelude::BixverseFloat;

///////////////////////////
// KNN label propagation //
///////////////////////////

/// Structure to store KNN graphs and do label propagation
///
/// ### Fields
///
/// * `offsets` - Stores the offsets for node i's neighbours
/// * `neighbours`- Flat array with neighbour indices
/// * `weights` - Normalised edge weights
#[derive(Debug, Clone)]
pub struct KnnLabPropGraph<T> {
    pub offsets: Vec<usize>,
    pub neighbours: Vec<usize>,
    pub weights: Vec<T>,
}

impl<T> KnnLabPropGraph<T>
where
    T: BixverseFloat,
{
    /// Generate the KnnLabelPropGraph
    ///
    /// ### Params
    ///
    /// * `edges` - edge list in form of [node_1, node_2, node_3, ...] which
    ///   indicates alternating pairs (node_1, node_2), etc in terms of edges.
    /// * `n_nodes` - Number of nodes in the graph
    ///
    /// ### Returns
    ///
    /// Self with the data stored in the structure.
    pub fn from_edge_list(edges: &[usize], n_nodes: usize) -> Self {
        // generate an adjaceny matrix for normalisation; could be faer
        let mut adj: Vec<Vec<(usize, T)>> = vec![vec![]; n_nodes];

        for chunk in edges.chunks(2) {
            let (u, v) = (chunk[0], chunk[1]);
            adj[u].push((v, T::one()));
            adj[v].push((u, T::one()));
        }

        for neighbours in &mut adj {
            let sum = T::from_usize(neighbours.len()).unwrap();
            for (_, w) in neighbours {
                *w /= sum;
            }
        }

        // conversion to CSR for better cache locality and look-ups
        let mut offsets: Vec<usize> = vec![0];
        let mut neighbours: Vec<usize> = Vec::with_capacity(edges.len());
        let mut weights: Vec<T> = Vec::with_capacity(edges.len());

        for node_neighbours in adj {
            for (neighbour, weight) in node_neighbours {
                neighbours.push(neighbour);
                weights.push(weight);
            }
            offsets.push(neighbours.len())
        }

        Self {
            offsets,
            neighbours,
            weights,
        }
    }

    /// Label spreading algorithm
    ///
    /// Function will spread the labels (one hot encoding for categorical data)
    /// over the graph. The input needs to be of structure:
    ///
    /// class 1 -> `[1.0, 0.0, 0.0, 0.0]`
    ///
    /// class 2 -> `[0.0, 1.0, 0.0, 0.0]`
    ///
    /// unlabelled -> `[0.0, 0.0, 0.0, 0.0]`
    ///
    /// ### Params
    ///
    /// * `labels` - One-hot encoded group membership. All zeroes == unlabelled.
    /// * `mask` - Boolean indicating which samples are unlabelled.
    /// * `alpha` - Controls the spreading. Usually between 0.9 to 0.95. Larger
    ///   values goes further labelling, smaller values are more conversative.
    ///
    /// ### Returns
    ///
    /// A Vec<Vec<f32>> with the probabilities of a given group being of that
    /// class.
    pub fn label_spreading(
        &self,
        labels: &[Vec<T>],
        mask: &[bool],
        alpha: T,
        iterations: usize,
        tolerance: T,
    ) -> Vec<Vec<T>> {
        let n = labels.len();
        let num_classes = labels[0].len();
        let mut y = labels.to_vec();
        let mut y_new = vec![vec![T::zero(); num_classes]; n];

        for _ in 0..iterations {
            y_new.par_iter_mut().enumerate().for_each(|(node, y_dist)| {
                let start = self.offsets[node];
                let end = self.offsets[node + 1];

                y_dist.fill(T::zero());
                for i in start..end {
                    let neighbor_dist = &y[self.neighbours[i]];
                    for c in 0..num_classes {
                        y_dist[c] += self.weights[i] * neighbor_dist[c];
                    }
                }
            });

            let max_change = y
                .par_iter_mut()
                .enumerate()
                .map(|(i, y_dist)| {
                    let mut max_diff = T::zero();

                    if mask[i] {
                        // unlabeled - pure propagation

                        for c in 0..num_classes {
                            max_diff = max_diff.max((y_new[i][c] - y_dist[c]).abs());
                            y_dist[c] = y_new[i][c];
                        }
                    } else {
                        // labeled - anchor to original

                        for c in 0..num_classes {
                            let new_val = alpha * labels[i][c] + (T::one() - alpha) * y_new[i][c];
                            max_diff = max_diff.max((new_val - y_dist[c]).abs());
                            y_dist[c] = new_val;
                        }
                    }

                    max_diff
                })
                .reduce(|| T::zero(), T::max);

            if max_change < tolerance {
                break;
            }
        }

        y
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_spreading_chain() {
        // Chain: 0 - 1 - 2 - 3
        let edges = vec![0, 1, 1, 2, 2, 3];
        let graph = KnnLabPropGraph::<f64>::from_edge_list(&edges, 4);

        // 2 classes.
        // Node 0 is class 0 [1.0, 0.0]
        // Node 3 is class 1 [0.0, 1.0]
        // Nodes 1, 2 are unlabeled [0.0, 0.0]
        let labels = vec![
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
        ];
        let mask = vec![false, true, true, false]; // true = unlabelled

        let probs = graph.label_spreading(&labels, &mask, 0.9, 100, 1e-4);

        // Node 1 is closer to Node 0, so it should have a higher probability for class 0
        assert!(probs[1][0] > probs[1][1]);
        // Node 2 is closer to Node 3, so it should have a higher probability for class 1
        assert!(probs[2][1] > probs[2][0]);

        // Anchors should remain strongly tied to their original classes
        assert!(probs[0][0] > 0.9);
        assert!(probs[3][1] > 0.9);
    }
}
