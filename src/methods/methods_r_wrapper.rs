use extendr_api::*;
use std::collections::BTreeMap;

use crate::core::mat_struct::NamedMatrix;
use crate::methods::cis_target::MotifEnrichment;
use crate::methods::dgrdl::DgrdlParams;
use crate::methods::ica::IcaParams;
use crate::prelude::*;

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
    pub fn from_r_list(r_list: List) -> DgrdlParams<f64> {
        let dgrdl_params = r_list.into_hashmap();

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

        DgrdlParams {
            sparsity,
            dict_size,
            alpha,
            beta,
            max_iter,
            k_neighbours,
            admm_iter,
            rho,
        }
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
    pub fn from_r_list(r_list: List) -> IcaParams<f64> {
        let ica_params = r_list.into_hashmap();

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

        IcaParams {
            maxit,
            alpha,
            tol,
            verbose,
        }
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
pub fn r_matrix_list_to_vec(matrix_list: List) -> Vec<(String, RArray<f64, [usize; 2]>)> {
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
///   name and the second the R matrix
///
/// ### Returns
///
/// A BTreeMap of `NamedMatrix` objects.
pub fn r_matrix_vec_to_named_matrices(
    matrix_vector: &[(String, RArray<f64, [usize; 2]>)],
) -> BTreeMap<String, NamedMatrix<'_, f64>> {
    let mut result = BTreeMap::new();
    for (name, matrix) in matrix_vector {
        let named_mat = NamedMatrix::<f64>::from_r_matrix(matrix);
        result.insert(name.clone(), named_mat);
    }

    result
}
