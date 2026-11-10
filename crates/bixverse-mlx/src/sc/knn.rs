//! MLX nearest neighbour search for single cell data, the twin of the wgpu
//! `bixverse_sc_gpu::knn_gpu`. Takes the same [`KnnParamsGpu`] and method
//! names; f32 only.

use ann_search_rs::prelude::CagraMlxSearchParams;
use ann_search_rs::{
    build_exhaustive_index_mlx, build_ivf_index_mlx, build_knn_graph_mlx,
    build_nndescent_index_mlx, query_exhaustive_index_mlx, query_exhaustive_index_mlx_self,
    query_ivf_index_mlx, query_ivf_index_mlx_self, query_nndescent_index_mlx,
    query_nndescent_index_mlx_self,
};
use bixverse_sc::sc_processing::knn::{to_true_distances, top_up_short_rows};
use bixverse_sc::sc_processing::knn_device::{
    KnnParamsGpu, KnnSearchGpu, cagra_beam, parse_knn_method_gpu,
};
use faer::MatRef;
use rayon::prelude::*;

use crate::prelude::*;

///////////
// Types //
///////////

/// Neighbour indices from an MLX query and, if requested, the distances.
pub type MlxKnnQueryResult = Result<(Vec<Vec<usize>>, Option<Vec<Vec<f32>>>), MlxErrors>;

/// Neighbour indices and distances from an MLX self-query, one row per cell.
pub type MlxKnnWithDist = Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), MlxErrors>;

/////////////
// Helpers //
/////////////

/// Resolve the method string, warning and falling back to the default on an
/// unrecognised one, as the wgpu dispatch does.
///
/// ### Params
///
/// * `params` - The [`KnnParamsGpu`] for this run.
///
/// ### Returns
///
/// The [`KnnSearchGpu`] to run.
fn resolve_method(params: &KnnParamsGpu) -> KnnSearchGpu {
    parse_knn_method_gpu(&params.knn_method).unwrap_or_else(|| {
        println!(
            "Unrecognised MLX kNN method provided: {:?}. Defaulting to exhaustive MLX.",
            params.knn_method
        );
        KnnSearchGpu::default()
    })
}

/// The MLX CAGRA query parameters, beam scaled to `k` by [`cagra_beam`].
///
/// ### Params
///
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `k_query` - Number of neighbours asked of the index, self included.
///
/// ### Returns
///
/// The [`CagraMlxSearchParams`] to hand to the query.
fn cagra_query_params_mlx(params: &KnnParamsGpu, k_query: usize) -> CagraMlxSearchParams {
    let (beam_width, max_beam_iters, n_entry_points) = cagra_beam(params, k_query);
    CagraMlxSearchParams::new(beam_width, max_beam_iters, n_entry_points, None)
}

/// Drop each query point from its own row and truncate to `k`.
///
/// ### Params
///
/// * `indices` - Rows from a self-query, up to `k + 1` long
/// * `distances` - Optional distances alongside
/// * `k` - Neighbours to keep per row
///
/// ### Returns
///
/// `(indices, distances)` with self removed, each row at most `k` long.
fn drop_self(
    indices: Vec<Vec<usize>>,
    distances: Option<Vec<Vec<f32>>>,
    k: usize,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    match distances {
        Some(distances) => {
            let (idx, dist): (Vec<Vec<usize>>, Vec<Vec<f32>>) = indices
                .into_par_iter()
                .zip(distances)
                .enumerate()
                .map(|(i, (idx, d))| {
                    idx.into_iter()
                        .zip(d)
                        .filter(|&(j, _)| j != i)
                        .take(k)
                        .unzip()
                })
                .unzip();
            (idx, Some(dist))
        }
        None => {
            let idx = indices
                .into_par_iter()
                .enumerate()
                .map(|(i, idx)| idx.into_iter().filter(|&j| j != i).take(k).collect())
                .collect();
            (idx, None)
        }
    }
}

/// Run the MLX self-query the parameters ask for.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours wanted per cell, self excluded.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - `0` silent, `1` normal, `2` detailed verbosity.
/// * `return_dist` - Also return the distances, in the index metric.
///
/// ### Returns
///
/// Rows with self removed, each `k` long (fewer only when there are not that
/// many other cells), and the distances if asked for.
fn self_query_mlx(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    seed: usize,
    verbose: usize,
    return_dist: bool,
) -> MlxKnnQueryResult {
    let level = parse_verbosity_level(verbose);
    let ann_verbosity = verbosity_from_flag(level.detailed_verbosity());
    let k_query = k + 1;

    let (indices, distances) = match resolve_method(params) {
        KnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_mlx(embd, &params.ann_dist, ann_verbosity)?;
            let (idx, dist) =
                query_exhaustive_index_mlx_self(&index, k_query, return_dist, ann_verbosity)?;
            drop_self(idx, dist, k)
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_mlx(
                embd,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                ann_verbosity,
            )?;
            let (idx, dist) = query_ivf_index_mlx_self(
                &index,
                k_query,
                params.n_probe,
                None,
                return_dist,
                ann_verbosity,
            )?;
            drop_self(idx, dist, k)
        }
        KnnSearchGpu::CagraGpu => {
            let index = build_nndescent_index_mlx(
                embd,
                &params.ann_dist,
                params.graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                ann_verbosity,
            )?;
            let (idx, dist) = query_nndescent_index_mlx_self(
                &index,
                k_query,
                Some(cagra_query_params_mlx(params, k_query)),
                return_dist,
                ann_verbosity,
            )?;
            drop_self(idx, dist, k)
        }
        KnnSearchGpu::NNDescentGraphGpu => {
            let graph = build_knn_graph_mlx(
                embd,
                &params.ann_dist,
                Some(k),
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                ann_verbosity,
            )?;
            let (mut idx, mut dist) = graph.extract_knn(None, false, return_dist);
            top_up_short_rows(
                embd,
                &mut idx,
                dist.as_deref_mut(),
                k,
                &params.ann_dist,
                level.normal_verbosity(),
            )?;
            (idx, dist)
        }
    };

    Ok((indices, distances))
}

//////////////
// Dispatch //
//////////////

/// MLX kNN indices, self excluded.
///
/// The twin of `dispatch_knn_gpu`. An unrecognised method string warns and
/// falls back to exhaustive rather than erroring.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours to return per cell.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - `0` silent, `1` normal, `2` detailed verbosity.
///
/// ### Returns
///
/// The kNN indices for the cells, each row `k` long.
///
/// ### Errors
///
/// * `AnnSearchRsError` if an index build or query fails.
pub fn dispatch_knn_mlx(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<usize>>, MlxErrors> {
    Ok(self_query_mlx(embd, k, params, seed, verbose, false)?.0)
}

/// MLX kNN with the distances, self excluded.
///
/// The twin of `knn_with_dist_gpu`. Euclidean distances come back as true L2,
/// not squared, via `to_true_distances`.
///
/// ### Params
///
/// * `embd` - Embedding matrix of `n_cells x features`.
/// * `k` - Number of neighbours to return per cell.
/// * `params` - The [`KnnParamsGpu`] for this run.
/// * `seed` - Seed for reproducibility of the index builds.
/// * `verbose` - `0` silent, `1` normal, `2` detailed verbosity.
///
/// ### Returns
///
/// The kNN indices and distances, one row per cell, sorted by distance.
///
/// ### Errors
///
/// As [`dispatch_knn_mlx`].
pub fn knn_with_dist_mlx(
    embd: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    seed: usize,
    verbose: usize,
) -> MlxKnnWithDist {
    let (indices, dist) = self_query_mlx(embd, k, params, seed, verbose, true)?;
    let mut dist = dist.expect("every MLX self-query returns distances when asked");
    to_true_distances(&mut dist, &params.ann_dist);
    Ok((indices, dist))
}

/// Build an MLX index on `reference` and query it with every row of `query`.
///
/// The cross-query twin of `dispatch_knn_query_gpu`. `"nndescent"` runs as
/// CAGRA, since a graph-only build cannot answer a cross-query.
///
/// ### Params
///
/// * `reference` - The data the index is built on (cells x features).
/// * `query` - The query points (cells x features).
/// * `k` - Neighbours to return per query point.
/// * `params` - The [`KnnParamsGpu`] for this run. `k` is ignored.
/// * `seed` - Random seed for the index build.
/// * `return_dist` - Return the distances as well, in the index metric
///   (squared for Euclidean).
/// * `verbose` - Detailed verbosity flag handed to the index build and query.
///
/// ### Returns
///
/// Reference-local neighbour indices, one row per query point, and the
/// distances if requested.
///
/// ### Errors
///
/// * `AnnSearchRsError` if an index build or query fails.
pub fn knn_query_mlx(
    reference: MatRef<f32>,
    query: MatRef<f32>,
    k: usize,
    params: &KnnParamsGpu,
    seed: usize,
    return_dist: bool,
    verbose: bool,
) -> MlxKnnQueryResult {
    let verbosity = verbosity_from_flag(verbose);

    let res = match resolve_method(params) {
        KnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_mlx(reference, &params.ann_dist, verbosity)?;
            query_exhaustive_index_mlx(query, &index, k, return_dist, verbosity)?
        }
        KnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_mlx(
                reference,
                params.n_list,
                None,
                &params.ann_dist,
                seed,
                verbosity,
            )?;
            query_ivf_index_mlx(
                query,
                &index,
                k,
                params.n_probe,
                None,
                return_dist,
                verbosity,
            )?
        }
        KnnSearchGpu::CagraGpu | KnnSearchGpu::NNDescentGraphGpu => {
            let index = build_nndescent_index_mlx(
                reference,
                &params.ann_dist,
                params.graph_k,
                params.k_build,
                None,
                params.n_tree,
                Some(params.delta),
                params.rho,
                params.refine_knn,
                seed,
                verbosity,
            )?;
            query_nndescent_index_mlx(
                query,
                &index,
                k,
                Some(cagra_query_params_mlx(params, k)),
                return_dist,
                verbosity,
            )?
        }
    };

    Ok(res)
}
