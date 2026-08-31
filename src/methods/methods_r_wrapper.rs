//! R wrapper functions for the various bioinformatics methods in this module

#[cfg(feature = "dge")]
use edge_rs::core::normalisation::{NormMethod, parse_norm_method};
#[cfg(feature = "dge")]
use edge_rs::glm::test::Tested;
use extendr_api::*;
use std::collections::{BTreeMap, HashMap};

use crate::core::mat_struct::NamedMatrix;
use crate::methods::cis_target::MotifEnrichment;
#[cfg(feature = "dge")]
use crate::methods::dge_bulk::EdgeRQlParams;
use crate::methods::dgrdl::DgrdlParams;
use crate::methods::ica::IcaParams;
use crate::methods::lda::metrics::LdaMetrics;
use crate::methods::lda::{LdaParams, LdaResult, LdaSweepResult, parse_lda_learning};
use crate::methods::nmf_hals::consensus::{ConsensusParams, parse_consensus_target};
use crate::methods::nmf_hals::{HalsOpts, parse_nmf_init};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Read an R scalar as a non-negative count, accepting either an integer or a
/// double.
///
/// R makes it easy to pass `10` (a double) where `10L` was meant, and
/// `extendr`'s `as_integer` only accepts INTSXP, so a type-strict read silently
/// falls back to the default. Negative values are floored at zero rather than
/// cast straight to `usize`, where `-1` becomes `usize::MAX` and turns a loop
/// bound into a hang.
///
/// ### Params
///
/// * `robj` - The R object to read.
///
/// ### Returns
///
/// The count, or `None` if the object is neither an integer nor a double.
fn robj_to_count(robj: &Robj) -> Option<usize> {
    robj_to_f64(robj).map(|v| if v > 0.0 { v.round() as usize } else { 0 })
}

/// Read an R scalar as an `f64`, accepting either an integer or a double.
///
/// ### Params
///
/// * `robj` - The R object to read.
///
/// ### Returns
///
/// The value, or `None` if the object is neither an integer nor a double.
fn robj_to_f64(robj: &Robj) -> Option<f64> {
    robj.as_real()
        .or_else(|| robj.as_integer().map(|v| v as f64))
}

///////////////
// CisTarget //
///////////////

/// Convert motif enrichments to R list format
///
/// ### Params
///
/// * `enrichments` - Slice of motif enrichment results
///
/// ### Returns
///
/// R list containing `motif_idx`, `nes`, `auc`, `rank_at_max`, `n_enriched`,
/// and `leading_edge`
pub fn motif_enrichments_to_r_list<T: BixverseFloat>(enrichments: &[MotifEnrichment<T>]) -> List {
    let mut result = List::new(6);

    let motif_idx: Vec<i32> = enrichments
        .iter()
        .map(|m| (m.motif_idx + 1) as i32)
        .collect();
    let nes: Vec<f64> = enrichments
        .iter()
        .map(|m| m.nes.to_f64().unwrap())
        .collect();
    let auc: Vec<f64> = enrichments
        .iter()
        .map(|m| m.auc.to_f64().unwrap())
        .collect();
    let rank_at_max: Vec<i32> = enrichments.iter().map(|m| m.rank_at_max as i32).collect();
    let n_enriched: Vec<i32> = enrichments.iter().map(|m| m.n_enriched as i32).collect();

    let mut leading_edge = List::new(enrichments.len());
    for (j, motif) in enrichments.iter().enumerate() {
        let genes: Vec<i32> = motif
            .enriched_gene_indices
            .iter()
            .map(|&idx| (idx + 1) as i32)
            .collect();
        leading_edge.set_elt(j, Robj::from(genes)).unwrap();
    }

    result.set_elt(0, Robj::from(motif_idx)).unwrap();
    result.set_elt(1, Robj::from(nes)).unwrap();
    result.set_elt(2, Robj::from(auc)).unwrap();
    result.set_elt(3, Robj::from(rank_at_max)).unwrap();
    result.set_elt(4, Robj::from(n_enriched)).unwrap();
    result.set_elt(5, Robj::from(leading_edge)).unwrap();

    result
        .set_names(&[
            "motif_idx",
            "nes",
            "auc",
            "rank_at_max",
            "n_enriched",
            "leading_edge",
        ])
        .unwrap();

    result
}

///////////
// Dgrdl //
///////////

impl<T> DgrdlParams<T>
where
    T: BixverseFloat,
{
    /// Generate the DGRDL parameters from an R list
    ///
    /// If values are not found, will use default values
    ///
    /// ### Params
    ///
    /// * `r_list` - The R list containing the parameters
    ///
    /// ### Returns
    ///
    /// The `DgrdlParams` structure based on the R list
    pub fn from_r_list(r_list: List) -> Result<DgrdlParams<f64>> {
        let dgrdl_params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let sparsity = dgrdl_params
            .get("sparsity")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let dict_size = dgrdl_params
            .get("dict_size")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let alpha = dgrdl_params
            .get("alpha")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0);
        let beta = dgrdl_params
            .get("beta")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0);
        let max_iter = dgrdl_params
            .get("max_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as usize;
        let k_neighbours = dgrdl_params
            .get("k_neighbours")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let admm_iter = dgrdl_params
            .get("admm_iter")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as usize;
        let rho = dgrdl_params
            .get("rho")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0);

        Ok(DgrdlParams {
            sparsity,
            dict_size,
            alpha,
            beta,
            max_iter,
            k_neighbours,
            admm_iter,
            rho,
        })
    }
}

/////////
// ICA //
/////////

impl<T: BixverseFloat> IcaParams<T> {
    /// Prepare ICA parameters from R List
    ///
    /// Takes in a R list and extracts the ICA parameters or uses sensible defaults.
    ///
    /// ### Params
    ///
    /// * `r_list` - R List with the ICA parameters.
    ///
    /// ### Returns
    ///
    /// `IcaParams` parameter structure.
    pub fn from_r_list(r_list: List) -> Result<IcaParams<f64>> {
        let ica_params: HashMap<&str, Robj> = r_list_to_map(r_list)?;

        let maxit = ica_params
            .get("maxit")
            .and_then(|v| v.as_integer())
            .unwrap_or(200) as usize;
        let alpha = ica_params
            .get("alpha")
            .and_then(|v| v.as_real())
            .unwrap_or(1.0);
        let tol = ica_params
            .get("max_tol")
            .and_then(|v| v.as_real())
            .unwrap_or(1e-4);
        let verbose = ica_params
            .get("verbose")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(IcaParams {
            maxit,
            alpha,
            tol,
            verbose,
        })
    }
}

/////////
// RBH //
/////////

/// Transforms a list of R matrices into a vector of R matrices
///
/// ### Params
///
/// * `matrix_list` - R List of matrices
///
/// ### Returns
///
/// A vector of tuples with the name of the list element and the R matrix.
pub fn r_matrix_list_to_vec(matrix_list: List) -> Vec<(String, RArray<f64, 2>)> {
    matrix_list
        .iter()
        .map(|(n, obj)| (n.to_string(), obj.as_matrix().unwrap()))
        .collect()
}

/// Take a vector of R matrices and generate a BTreeMap of NamedMatrices
///
/// ### Params
///
/// * `matrix_vector` - Slice of tuples with the first element representing the
///   name and the second the R matrix.
///
/// ### Returns
///
/// A BTreeMap of `NamedMatrix` objects.
pub fn r_matrix_vec_to_named_matrices(
    matrix_vector: &[(String, RArray<f64, 2>)],
) -> BTreeMap<String, NamedMatrix<'_, f64>> {
    let mut result = BTreeMap::new();
    for (name, matrix) in matrix_vector {
        let named_mat = NamedMatrix::<f64>::from_r_matrix(matrix);
        result.insert(name.clone(), named_mat);
    }

    result
}

/////////////
// HalsOpt //
/////////////

impl<T> HalsOpts<T>
where
    T: BixverseFloat,
{
    /// Generate [HalsOpts] from R list
    ///
    /// ### Params
    ///
    /// * `r_list` - The List from which to extract the parameters. If
    ///   parameters are not found, defaults to sensible defaults.
    /// * `seed` - Seed for random initialisation of NMF
    ///
    /// ### Returns
    ///
    /// The [HalsOpts]
    pub fn from_r_list(r_list: List, seed: usize) -> Result<HalsOpts<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: HalsOpts<T> = HalsOpts::default();

        let max_iter = params
            .get("max_iter")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_iter);

        let check_every = params
            .get("check_every")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(defaults.check_every);

        let tol = params
            .get("tol")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.tol);

        let eps = params
            .get("eps")
            .and_then(|v| v.as_real())
            .map(|v| T::from_f64(v).unwrap())
            .unwrap_or(defaults.eps);

        let nmf_init = params
            .get("nmf_init")
            .and_then(|v| v.as_str())
            .and_then(|v| parse_nmf_init(v, seed))
            .unwrap_or(defaults.init);

        Ok(HalsOpts::new(max_iter, tol, eps, check_every, nmf_init))
    }
}

/////////////////////
// ConsensusParams //
/////////////////////

impl<T> ConsensusParams<T>
where
    T: BixverseFloat,
{
    /// Generate [ConsensusParams] from R list
    ///
    /// A `density_threshold` at or above 2 is taken as "no filtering", since
    /// cosine distance cannot exceed 2. That gives R a single numeric knob
    /// rather than a numeric plus a toggle. `n_neighbours = 0` means
    /// "pick for me", the same as omitting it.
    ///
    /// Every numeric field accepts an R integer or a double, because
    /// `density_threshold = 2L` silently failing to disable the filter is the
    /// exact opposite of what the caller asked for. Negative counts are floored
    /// at zero rather than wrapping into a huge `usize`.
    ///
    /// ### Params
    ///
    /// * `r_list` - The List from which to extract the parameters. If
    ///   parameters are not found, defaults to sensible defaults.
    ///
    /// ### Returns
    ///
    /// The [ConsensusParams]
    pub fn from_r_list(r_list: List) -> Result<ConsensusParams<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: ConsensusParams<T> = ConsensusParams::default();

        let target = params
            .get("consensus_target")
            .and_then(|v| v.as_str())
            .and_then(parse_consensus_target)
            .unwrap_or(defaults.target);

        // Zero is "pick for me" here and in `resolve_n_neighbours`, so the two
        // sides of the boundary agree on what it means.
        let n_neighbours = params
            .get("n_neighbours")
            .and_then(robj_to_count)
            .and_then(|v| if v > 0 { Some(v) } else { None })
            .or(defaults.n_neighbours);

        let density_threshold = match params.get("density_threshold").and_then(robj_to_f64) {
            Some(v) if v >= 2.0 => None,
            Some(v) => Some(T::from_f64(v).unwrap()),
            None => defaults.density_threshold,
        };

        let kmeans_iters = params
            .get("kmeans_iters")
            .and_then(robj_to_count)
            .unwrap_or(defaults.kmeans_iters);

        let kmeans_n_init = params
            .get("kmeans_n_init")
            .and_then(robj_to_count)
            .unwrap_or(defaults.kmeans_n_init);

        let seed = params
            .get("consensus_seed")
            .and_then(robj_to_count)
            .map(|v| v as u64)
            .unwrap_or(defaults.seed);

        Ok(ConsensusParams::new(
            target,
            n_neighbours,
            density_threshold,
            kmeans_iters,
            kmeans_n_init,
            seed,
        ))
    }
}

///////////////
// LdaParams //
///////////////

impl<T> LdaParams<T>
where
    T: BixverseFloat,
{
    /// Generate [LdaParams] from R list
    ///
    /// Unrecognised strings for `learning` fall back to the default rather than
    /// erroring, matching [HalsOpts::from_r_list]. `batch_size` and `n_epochs`
    /// are only read when `learning` resolves to the online variant.
    ///
    /// ### Params
    ///
    /// * `r_list` - The List from which to extract the parameters. If
    ///   parameters are not found, defaults to sensible defaults.
    /// * `seed` - Seed for the variational initialisation.
    ///
    /// ### Returns
    ///
    /// The [LdaParams]
    pub fn from_r_list(r_list: List, seed: usize) -> Result<LdaParams<T>> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults: LdaParams<T> = LdaParams::default();

        let read_f = |key: &str, fallback: T| -> T {
            params
                .get(key)
                .and_then(robj_to_f64)
                .and_then(T::from_f64)
                .unwrap_or(fallback)
        };
        let read_usize = |key: &str, fallback: usize| -> usize {
            params.get(key).and_then(robj_to_count).unwrap_or(fallback)
        };
        let read_bool = |key: &str, fallback: bool| -> bool {
            params
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or(fallback)
        };

        let batch_size = read_usize("batch_size", 1024);
        let n_epochs = read_usize("n_epochs", 10);
        let learning = params
            .get("learning")
            .and_then(|v| v.as_str())
            .and_then(|v| parse_lda_learning(v, batch_size, n_epochs))
            .unwrap_or(defaults.learning);

        Ok(LdaParams::new(
            read_f("alpha", defaults.alpha),
            read_bool("alpha_by_topic", defaults.alpha_by_topic),
            read_f("eta", defaults.eta),
            read_bool("eta_by_topic", defaults.eta_by_topic),
            read_usize("max_iter", defaults.max_iter),
            read_f("tol", defaults.tol),
            read_usize("inner_max_iter", defaults.inner_max_iter),
            read_f("inner_tol", defaults.inner_tol),
            read_usize("check_every", defaults.check_every),
            learning,
            seed as u64,
        ))
    }
}

/// Convert a fitted LDA model to R list format
///
/// ### Params
///
/// * `model` - The fitted model.
///
/// ### Returns
///
/// R list containing `cell_topic` (topics x cells), `topic_region` (regions x
/// topics), `bound`, `perplexity`, `n_iter` and `converged`.
pub fn lda_result_to_r_list<T>(model: &LdaResult<T>) -> extendr_api::Result<List>
where
    T: BixverseFloat + FaerRType,
{
    let mut res = List::new(6);
    res.set_elt(0, faer_to_r_matrix(model.cell_topic.as_ref()).into())?;
    res.set_elt(1, faer_to_r_matrix(model.topic_region.as_ref()).into())?;
    res.set_elt(2, model.bound.to_f64().into_robj())?;
    res.set_elt(3, model.perplexity.to_f64().into_robj())?;
    res.set_elt(4, (model.n_iter as i32).into_robj())?;
    res.set_elt(5, model.converged.into_robj())?;
    res.set_names([
        "cell_topic",
        "topic_region",
        "bound",
        "perplexity",
        "n_iter",
        "converged",
    ])?;
    Ok(res)
}

/// Convert LDA model selection metrics to R list format
///
/// ### Params
///
/// * `metrics` - Metrics for one fitted model.
///
/// ### Returns
///
/// R list containing `arun_2010`, `cao_juan_2009`, `mimno_2011`,
/// `coherence_per_topic`, `bound` and `perplexity`.
pub fn lda_metrics_to_r_list<T>(metrics: &LdaMetrics<T>) -> extendr_api::Result<List>
where
    T: BixverseFloat,
{
    let coherence: Vec<f64> = metrics
        .coherence_per_topic
        .iter()
        .map(|v| v.to_f64().unwrap_or(f64::NAN))
        .collect();

    let mut res = List::new(6);
    res.set_elt(0, metrics.arun_2010.to_f64().into_robj())?;
    res.set_elt(1, metrics.cao_juan_2009.to_f64().into_robj())?;
    res.set_elt(2, metrics.mimno_2011.to_f64().into_robj())?;
    res.set_elt(3, coherence.into_robj())?;
    res.set_elt(4, metrics.bound.to_f64().into_robj())?;
    res.set_elt(5, metrics.perplexity.to_f64().into_robj())?;
    res.set_names([
        "arun_2010",
        "cao_juan_2009",
        "mimno_2011",
        "coherence_per_topic",
        "bound",
        "perplexity",
    ])?;
    Ok(res)
}

/// Convert an LDA topic-count sweep to R list format
///
/// ### Params
///
/// * `sweep` - The sweep result.
///
/// ### Returns
///
/// R list containing `k` (the topic counts tried), `models`, `metrics`,
/// `combined_score` and `best_k`. Entries excluded from selection by the
/// topic-count floor carry `NA` in `combined_score`.
pub fn lda_sweep_to_r_list<T>(sweep: &LdaSweepResult<T>) -> extendr_api::Result<List>
where
    T: BixverseFloat + FaerRType,
{
    let ks: Vec<i32> = sweep.entries.iter().map(|e| e.k as i32).collect();
    let models: Vec<Robj> = sweep
        .entries
        .iter()
        .map(|e| lda_result_to_r_list(&e.model).map(|l| l.into_robj()))
        .collect::<extendr_api::Result<_>>()?;
    let metrics: Vec<Robj> = sweep
        .entries
        .iter()
        .map(|e| lda_metrics_to_r_list(&e.metrics).map(|l| l.into_robj()))
        .collect::<extendr_api::Result<_>>()?;
    let scores: Vec<f64> = sweep
        .combined_score
        .iter()
        .map(|v| v.to_f64().unwrap_or(f64::NAN))
        .collect();

    let mut res = List::new(5);
    res.set_elt(0, ks.into_robj())?;
    res.set_elt(1, List::from_values(models).into_robj())?;
    res.set_elt(2, List::from_values(metrics).into_robj())?;
    res.set_elt(3, scores.into_robj())?;
    res.set_elt(4, (sweep.best_k as i32).into_robj())?;
    res.set_names(["k", "models", "metrics", "combined_score", "best_k"])?;
    Ok(res)
}

///////////////////
// edgeR bulk DE //
///////////////////

/// R-list parsing for [Tested].
///
/// A trait rather than an inherent impl because [Tested] is defined in
/// `edge-rs`.
#[cfg(feature = "dge")]
pub trait TestedFromR: Sized {
    /// Parse the [Tested] from a list
    ///
    /// Expects either `coef`, zero-based design columns to drop from the null,
    /// or `contrast`, column-major weights with `n_contrasts` columns. There is
    /// no default: which effect to report is the question the caller came to
    /// ask.
    ///
    /// ### Params
    ///
    /// * `params` - The flattened R list to parse
    ///
    /// ### Returns
    ///
    /// The [Tested], or an error if neither key is present.
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self>;
}

/// [TestedFromR] implementation
#[cfg(feature = "dge")]
impl TestedFromR for Tested {
    fn from_r_map(params: &HashMap<&str, Robj>) -> Result<Self> {
        if let Some(values) = params.get("contrast").and_then(|v| v.as_real_vector()) {
            let n_contrasts = params
                .get("n_contrasts")
                .and_then(|v| v.as_integer())
                .map(|v| v as usize)
                .unwrap_or(1);
            return Ok(Tested::Contrast {
                values,
                n_contrasts,
            });
        }
        // `c(2)` is a double in R and `c(2L)` is not, and a type-strict read
        // would take one of them for an absent key.
        let coef: Option<Vec<usize>> = params.get("coef").and_then(|v| {
            v.as_integer_vector()
                .map(|xs| xs.into_iter().map(|x| x as usize).collect())
                .or_else(|| {
                    v.as_real_vector()
                        .map(|xs| xs.into_iter().map(|x| x as usize).collect())
                })
        });

        match coef {
            Some(coef) => Ok(Tested::Coef(coef)),
            None => Err(Error::Other(
                "the edgeR test needs either `coef` or `contrast` to know what to test".into(),
            )),
        }
    }
}

#[cfg(feature = "dge")]
impl EdgeRQlParams {
    /// Generate EdgeRQlParams from an R list
    ///
    /// Everything falls back to edgeR's own defaults. `norm_method` accepts
    /// edgeR's spellings, `"none"` included, and errors on anything else rather
    /// than normalising with something other than what was asked for.
    ///
    /// ### Params
    ///
    /// * `r_list` - The list with the edgeR parameters.
    ///
    /// ### Returns
    ///
    /// The `EdgeRQlParams` with all parameters set.
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let params: HashMap<&str, Robj> = r_list_to_map(r_list)?;
        let defaults = Self::default();

        let norm_method: NormMethod = match params.get("norm_method").and_then(|v| v.as_str()) {
            Some(s) => parse_norm_method(s)
                .ok_or_else(|| Error::Other(format!("Invalid normalisation method: {}", s)))?,
            None => defaults.norm_method,
        };

        Ok(Self {
            norm_method,
            filter: params
                .get("filter")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.filter),
            min_mean: params
                .get("min_mean")
                .and_then(|v| v.as_real())
                .unwrap_or(defaults.min_mean),
            robust: params
                .get("robust")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.robust),
            legacy: params
                .get("legacy")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.legacy),
        })
    }
}
