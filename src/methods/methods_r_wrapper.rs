use extendr_api::{Attributes, List, Robj};

use crate::methods::cis_target::MotifEnrichment;
use crate::prelude::*;

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
