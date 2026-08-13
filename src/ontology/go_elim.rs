//! Implementation of the elimination method for improved gene ontology-based
//! enrichment. See Alexa, et al., Bioinformatics, 2006

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::collections::BTreeMap;

use crate::core::math::stats::*;
use crate::enrichment::gsea::*;
use crate::enrichment::oae::*;
use crate::prelude::*;

///////////////////////
// Types & Structure //
///////////////////////

/// Type alias for the go identifier to gene Hashmap
pub type GeneMap = FxHashMap<String, FxHashSet<String>>;

/// Type alias for the ancestor to go identifier HashMap
pub type AncestorMap = FxHashMap<String, Vec<String>>;

/// Type alias for the ontology level to go identifier HashMap
pub type LevelMap = FxHashMap<String, Vec<String>>;

/// Type alias for intermediary results
///
/// ### Fields
///
/// * `0` - ES scores
/// * `1` - Size
/// * `2` - Indices of the leading edge genes.
type GoIntermediaryRes<T> = (T, usize, Vec<i32>);

/// Return structure of the `process_ontology_level()` ontology function.
#[derive(Clone, Debug)]
pub struct GoElimLevelResults<T> {
    /// GO term identifiers.
    pub go_ids: Vec<String>,
    /// P-values for the terms
    pub pvals: Vec<T>,
    /// Odds ratios for the terms.
    pub odds_ratios: Vec<T>,
    /// Number of intersecting genes with the target gene set
    pub hits: Vec<usize>,
    /// Length of the gene ontology term after elimination.
    pub gene_set_lengths: Vec<usize>,
}

/// Final return structure after filtering
#[derive(Clone, Debug)]
pub struct GoElimFinalResults<T> {
    /// GO term identifiers.
    pub go_ids: Vec<String>,
    /// P-values for the terms.
    pub pvals: Vec<T>,
    /// Calculated FDRs.
    pub fdr: Vec<T>,
    /// Odds ratios for the terms.
    pub odds_ratios: Vec<T>,
    /// Number of intersecting genes with the target gene set
    pub hits: Vec<usize>,
    /// Length of the gene set after elimination.
    pub gs_length: Vec<usize>,
}

/// Return structure of the `process_ontology_level_fgsea_simple()` ontology
/// function.
#[derive(Clone, Debug)]
pub struct GoElimLevelResultsGsea<T> {
    /// GO term identifiers.
    pub go_ids: Vec<String>,
    /// Enrichment Scores for the terms.
    pub es: Vec<T>,
    /// Normalised enrichment Scores for the terms.
    pub nes: Vec<Option<T>>,
    /// The sizes of the terms.
    pub size: Vec<usize>,
    /// The P-value for the terms.
    pub pvals: Vec<T>,
    /// The number of permuted values that were larger (or smaller).
    pub n_more_extreme: Vec<usize>,
    /// The number of permuted values that were ≥ 0.
    pub ge_zero: Vec<usize>,
    /// The number of permuted values that were ≤ 0.
    pub le_zero: Vec<usize>,
    /// Indices of the leading edge genes of the term.
    pub leading_edge: Vec<Vec<i32>>,
}

///////////////////////
// GO data structure //
///////////////////////

/// Structure for the GeneOntology data
#[derive(Clone, Debug)]
pub struct GeneOntology<'a> {
    /// A HashMap with the term to gene associations
    pub go_to_gene: GeneMap,
    /// A reference to the HashMap with the term to ancestor associations.
    pub ancestors: &'a AncestorMap,
    /// A reference to the  HashMap with the ontology level to term associations.
    pub levels: &'a LevelMap,
}

impl<'a> GeneOntology<'a> {
    /// Create a new GeneOntology Structure
    ///
    /// ### Params
    ///
    /// * `gene_map` - The HashMap with the term to gene associations.
    /// * `ancestors` - The HashMap with the term to ancestor associations.
    /// * `levels` - The HashMap with the level to term associations.
    ///
    /// ### Returns
    ///
    /// Initialised structure
    pub fn new(gene_map: GeneMap, ancestor_map: &'a AncestorMap, levels_map: &'a LevelMap) -> Self {
        GeneOntology {
            go_to_gene: gene_map,
            ancestors: ancestor_map,
            levels: levels_map,
        }
    }

    /// Returns the ancestors of a given gene ontology term identifier
    ///
    /// ### Params
    ///
    /// * `id` - The identifier of the term for which to retrieve the ancestors
    ///
    /// ### Returns
    ///
    /// The ancestors of that term.
    pub fn get_ancestors(&self, id: &String) -> Option<&Vec<String>> {
        self.ancestors.get(id)
    }

    /// Returns the terms for a given level identifier
    ///
    /// ### Params
    ///
    /// * `id` - The level identifier for which to return the GO terms.
    ///
    /// ### Returns
    ///
    /// The GO terms at this level.
    pub fn get_level_ids(&self, id: &String) -> Option<&Vec<String>> {
        self.levels.get(id)
    }

    /// Eliminate genes in a subset of GO terms
    ///
    /// This function implements the elimination logic. For a set of provided
    /// terms it will remove the genes (if they are part of the term) for the
    /// provided GO terms.
    ///
    /// ### Params
    ///
    /// * `ids` - The GO terms in which to remove the genes.
    /// * `genes_to_remove` - HashSet of the genes to remove.
    pub fn remove_genes(&mut self, ids: &[String], genes_to_remove: &FxHashSet<String>) {
        for id in ids.iter() {
            if let Some(gene_set) = self.go_to_gene.get_mut(id) {
                gene_set.retain(|gene| !genes_to_remove.contains(gene));
            }
        }
    }

    /// Get the term to gene associations
    ///
    /// ### Param
    ///
    /// * `ids` - The GO terms for which to return the genes
    ///
    /// ### Returns
    ///
    /// A HashMap with the genes as HashSets.
    pub fn get_genes_list(&self, ids: &[String]) -> FxHashMap<String, &FxHashSet<String>> {
        let mut to_ret = FxHashMap::with_capacity_and_hasher(ids.len(), FxBuildHasher);

        for id in ids.iter() {
            if let Some(gene_set) = self.go_to_gene.get(id) {
                to_ret.insert(id.clone(), gene_set);
            }
        }

        to_ret
    }

    /// Return the genes for a specific term
    ///
    /// ### Params
    ///
    /// * `id` - The GO term for which to return the genes
    ///
    /// ### Returns
    ///
    /// The HashSet with the genes.
    pub fn get_genes(&self, id: &String) -> Option<&FxHashSet<String>> {
        self.go_to_gene.get(id)
    }
}

//////////////////////////////
// GO permutation structure //
//////////////////////////////

/// Structure to hold random permutations for the continuous (fgsea)
#[derive(Clone, Debug)]
pub struct GeneOntologyRandomPerm<'a, T> {
    /// A reference to the vector of vectors with the permutation ES.
    pub random_perm: &'a Vec<Vec<T>>,
}

impl<'a, T> GeneOntologyRandomPerm<'a, T>
where
    T: BixverseFloat,
{
    /// Initialise the structure
    ///
    /// ### Params
    ///
    /// * `perm_es` - A vector of vectors with the permutation results
    ///
    /// ### Returns
    ///
    /// Initialised structure
    pub fn new(perm_es: &'a Vec<Vec<T>>) -> Self {
        GeneOntologyRandomPerm {
            random_perm: perm_es,
        }
    }

    /// Return GseaResults (simple)
    ///
    /// ### Params
    ///
    /// * `pathway_scores` - The pathway scores for which to estimate the permutation
    ///   based stats
    /// * `pathway_sizes` - The corresponding pathway sizes.
    ///
    /// ### Returns
    ///
    /// A `GseaResults` structure with all of the statistics.
    pub fn get_gsea_res_simple<'b>(
        &self,
        pathway_scores: &'b [T],
        pathway_sizes: &'b [usize],
    ) -> GseaResults<'b, T> {
        // Dual lifetimes fun...
        let gsea_batch_res: GseaBatchResults<T> =
            calc_gsea_stats_wrapper(pathway_scores, pathway_sizes, self.random_perm);

        let gsea_res: GseaResults<'_, T> =
            calculate_nes_es_pval(pathway_scores, pathway_sizes, &gsea_batch_res);

        gsea_res
    }
}

///////////////
// Functions //
///////////////

/////////////
// Helpers //
/////////////

/// Finalise the GO enrichment results from a hypergeometric test
///
/// ### Params
///
/// * `go_res` - Slice of `GoElimLevelResults` structures
/// * `min_overlap` - Optional minimum overlap.
/// * `fdr_threshold` - Optional fdr threshold.
///
/// ### Returns
///
/// A `GoElimFinalResults` results structure
pub fn finalise_go_res<T>(
    go_res: &[GoElimLevelResults<T>],
    min_overlap: Option<usize>,
    fdr_threshold: Option<T>,
) -> GoElimFinalResults<T>
where
    T: BixverseFloat,
{
    let n = go_res.iter().map(|x| x.go_ids.len()).sum::<usize>();

    let mut go_ids: Vec<Vec<String>> = Vec::with_capacity(n);
    let mut pvals: Vec<Vec<T>> = Vec::with_capacity(n);
    let mut hits: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut odds_ratios: Vec<Vec<T>> = Vec::with_capacity(n);
    let mut gs_lengths: Vec<Vec<usize>> = Vec::with_capacity(n);

    for res in go_res {
        go_ids.push(res.go_ids.clone());
        pvals.push(res.pvals.clone());
        hits.push(res.hits.clone());
        odds_ratios.push(res.odds_ratios.clone());
        gs_lengths.push(res.gene_set_lengths.clone());
    }

    let go_ids = flatten_vector(go_ids);
    let pvals = flatten_vector(pvals);
    let fdr = calc_fdr(&pvals);
    let hits = flatten_vector(hits);
    let odds_ratios = flatten_vector(odds_ratios);
    let gs_lengths = flatten_vector(gs_lengths);

    let to_keep: Vec<usize> = (0..n)
        .filter(|i| {
            if let Some(min_overlap) = min_overlap
                && hits[*i] < min_overlap
            {
                return false;
            }

            if let Some(fdr_threshold) = fdr_threshold
                && fdr[*i] > fdr_threshold
            {
                return false;
            }
            true
        })
        .collect();

    GoElimFinalResults {
        go_ids: to_keep.iter().map(|i| go_ids[*i].clone()).collect(),
        pvals: to_keep.iter().map(|i| pvals[*i]).collect(),
        fdr: to_keep.iter().map(|i| fdr[*i]).collect(),
        odds_ratios: to_keep.iter().map(|i| odds_ratios[*i]).collect(),
        hits: to_keep.iter().map(|i| hits[*i]).collect(),
        gs_length: to_keep.iter().map(|i| gs_lengths[*i]).collect(),
    }
}

/////////////////////
// Hypergeom tests //
/////////////////////

/// Process a given ontology level (hypergeometric test)
///
/// ### Params
///
/// * `target_set` - The HashSet of target genes.
/// * `level` - The ontology level to process.
/// * `go_obj` The `GeneOntology` structure with the needed data.
/// * `min_genes` - Minimum number of genes in a given term.
/// * `gene_universe_length` - Size of the gene universe
/// * `elim_threshold` - Elimination threshold. Below that threshold the genes
///   from that term are eliminated from its ancestors
/// * `debug` - Shall debug messages be printed.
///
/// ### Return
///
/// The `GoElimLevelResults` with the calculated statistics.
pub fn process_ontology_level<T>(
    target_set: &FxHashSet<String>,
    level: &String,
    go_obj: &mut GeneOntology,
    min_genes: usize,
    gene_universe_length: usize,
    elim_threshold: T,
) -> GoElimLevelResults<T>
where
    T: BixverseFloat,
{
    // Get the identifiers of that level and clean everything up
    let binding: Vec<String> = Vec::new();

    let level_ids = go_obj.get_level_ids(level).unwrap_or(&binding);
    let level_data = go_obj.get_genes_list(level_ids);

    // Filter data based on minimum gene requirement
    let level_data_final: FxHashMap<_, _> = level_data
        .iter()
        .filter(|(_, value)| value.len() >= min_genes)
        .map(|(key, value)| (key.clone(), value))
        .collect();

    let trials = target_set.len();
    let size = level_data_final.len();

    let mut go_ids = Vec::with_capacity(size);
    let mut hits_vec = Vec::with_capacity(size);
    let mut pvals = Vec::with_capacity(size);
    let mut odds_ratios = Vec::with_capacity(size);
    let mut gene_set_lengths = Vec::with_capacity(size);

    for (key, value) in level_data_final {
        let gene_set_length = value.len();
        let hits = target_set.intersection(value).count();
        // A single hit gives q = 0, which is P(X >= 1) and perfectly meaningful;
        // only zero hits is the degenerate case.
        let pval: T = if hits > 0 {
            hypergeom_pval(
                hits - 1,
                gene_set_length,
                gene_universe_length - gene_set_length,
                trials,
            )
        } else {
            T::one()
        };
        let odds_ratio = hypergeom_odds_ratio(
            hits,
            gene_set_length - hits,
            trials - hits,
            // Grouped so the addition lands before the subtractions, otherwise
            // `N - m - k` underflows on a large target set.
            (gene_universe_length + hits) - gene_set_length - trials,
        );
        go_ids.push(key.clone());
        hits_vec.push(hits);
        pvals.push(pval);
        odds_ratios.push(odds_ratio);
        gene_set_lengths.push(gene_set_length);
    }

    let res = GoElimLevelResults {
        go_ids,
        pvals,
        odds_ratios,
        hits: hits_vec,
        gene_set_lengths,
    };

    // Identify the GO terms were to apply the elimination on (if any)
    let go_to_remove: &Vec<_> = &res
        .go_ids
        .iter()
        .zip(&res.pvals)
        .filter(|(_, pval)| pval <= &&elim_threshold)
        .map(|(string, _)| string.clone())
        .collect();

    for term in go_to_remove.iter() {
        if let Some(ancestors) = go_obj.get_ancestors(term) {
            let ancestors_final: Vec<String> = ancestors.to_vec();
            if let Some(genes_to_remove) = go_obj.get_genes(term) {
                let genes_to_remove = genes_to_remove.clone();

                go_obj.remove_genes(&ancestors_final, &genes_to_remove);
            }
        }
    }

    res
}

/////////////////////
// Continuous test //
/////////////////////

/// Process a given ontology level (simple fgsea)
///
/// ### Params
///
/// * `stats` - The gene level statistic.
/// * `stat_name_indices` - A hashmap linking genes to index positions in stats.
/// * `level` - The ontology level to process.
/// * `go_obj` The `GeneOntology` structure with the needed data.
/// * `go_random_perms` - The `GeneOntologyRandomPerm` with the random permutations.
/// * `gsea_params` - The GSEA parameter.
/// * `elim_threshold` - Elimination threshold. Below that threshold the genes
///   from that term are eliminated from its ancestors
///
/// ### Return
///
/// The `GoElimLevelResultsGsea` with the calculated statistics.
#[allow(clippy::too_many_arguments)]
pub fn process_ontology_level_fgsea_simple<T>(
    stats: &[T],
    stat_name_indices: &FxHashMap<&String, usize>,
    level: &String,
    go_obj: &mut GeneOntology,
    go_random_perms: &GeneOntologyRandomPerm<T>,
    gsea_params: &GseaParams<T>,
    elim_threshold: T,
) -> GoElimLevelResultsGsea<T>
where
    T: BixverseFloat,
{
    // Get the identfiers of that level and clean everything up
    let binding: Vec<String> = Vec::new();
    let go_ids = go_obj.get_level_ids(level).unwrap_or(&binding);

    // BTreeMap to make sure the order is determistic
    let mut level_data_es: BTreeMap<String, GoIntermediaryRes<T>> = BTreeMap::new();

    for go_id in go_ids {
        if let Some(genes) = go_obj.get_genes(go_id)
            && genes.len() >= gsea_params.min_size
            && genes.len() <= gsea_params.max_size
        {
            // Convert gene names to indices in one step
            let mut indices: Vec<i32> = genes
                .iter()
                .filter_map(|gene| stat_name_indices.get(gene).map(|&i| i as i32))
                .collect();

            indices.sort();

            if !indices.is_empty() {
                let es_res =
                    calc_gsea_stats(stats, &indices, gsea_params.gsea_param, true, false, false);
                let size = indices.len();
                level_data_es.insert(go_id.clone(), (es_res.es, size, es_res.leading_edge));
            }
        }
    }

    let mut pathway_scores: Vec<T> = Vec::with_capacity(level_data_es.len());
    let mut pathway_sizes: Vec<usize> = Vec::with_capacity(level_data_es.len());
    let mut leading_edge_indices: Vec<Vec<i32>> = Vec::with_capacity(level_data_es.len());

    for v in level_data_es.values() {
        pathway_scores.push(v.0);
        pathway_sizes.push(v.1);
        leading_edge_indices.push(v.2.clone());
    }

    let level_res: GseaResults<'_, T> =
        go_random_perms.get_gsea_res_simple(&pathway_scores, &pathway_sizes);

    let go_to_remove: Vec<&String> = level_res
        .pvals
        .iter()
        .zip(level_data_es.keys())
        .filter(|(pval, _)| **pval <= elim_threshold) // Fixed double reference and typo
        .map(|(_, go_id)| go_id)
        .collect();

    for term in go_to_remove.iter() {
        if let Some(ancestors) = go_obj.get_ancestors(term) {
            let ancestors_final: Vec<String> = ancestors.to_vec();
            if let Some(genes_to_remove) = go_obj.get_genes(term) {
                let genes_to_remove = genes_to_remove.clone();

                go_obj.remove_genes(&ancestors_final, &genes_to_remove);
            }
        }
    }

    GoElimLevelResultsGsea {
        go_ids: level_data_es.into_keys().collect(),
        es: pathway_scores.clone(),
        nes: level_res.nes,
        size: pathway_sizes.clone(),
        pvals: level_res.pvals,
        n_more_extreme: level_res.n_more_extreme,
        ge_zero: level_res.ge_zero,
        le_zero: level_res.le_zero,
        leading_edge: leading_edge_indices,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn genes(range: std::ops::RangeInclusive<usize>) -> FxHashSet<String> {
        range.map(|i| format!("g{i}")).collect()
    }

    /// `GO:B` (g1..g5) sits one level below `GO:A` (g1..g10), so eliminating
    /// `GO:B` must strip its genes from `GO:A`.
    fn two_level_ontology() -> (GeneMap, AncestorMap, LevelMap) {
        let mut gene_map = GeneMap::default();
        gene_map.insert("GO:A".to_string(), genes(1..=10));
        gene_map.insert("GO:B".to_string(), genes(1..=5));

        let mut ancestors = AncestorMap::default();
        ancestors.insert("GO:B".to_string(), vec!["GO:A".to_string()]);

        let mut levels = LevelMap::default();
        levels.insert("1".to_string(), vec!["GO:A".to_string()]);
        levels.insert("2".to_string(), vec!["GO:B".to_string()]);

        (gene_map, ancestors, levels)
    }

    /// The accessors are plain map lookups and must tolerate unknown ids.
    #[test]
    fn gene_ontology_accessors_ignore_unknown_terms() {
        let (gene_map, ancestors, levels) = two_level_ontology();
        let mut go = GeneOntology::new(gene_map, &ancestors, &levels);

        assert_eq!(go.get_genes(&"GO:A".to_string()).unwrap().len(), 10);
        assert!(go.get_genes(&"GO:MISSING".to_string()).is_none());
        assert_eq!(go.get_level_ids(&"2".to_string()).unwrap(), &vec!["GO:B"]);
        assert_eq!(
            go.get_genes_list(&["GO:A".to_string(), "GO:MISSING".to_string()])
                .len(),
            1
        );

        go.remove_genes(&["GO:A".to_string()], &genes(1..=2));
        assert_eq!(go.get_genes(&"GO:A".to_string()).unwrap().len(), 8);

        // A term with no entry must not panic.
        go.remove_genes(&["GO:MISSING".to_string()], &genes(1..=2));
    }

    /// The elim algorithm itself. Processing the lower level first must strip
    /// `GO:B`'s five genes out of its ancestor, so the same target set then
    /// scores zero hits against `GO:A`. Every assertion here is a whole number.
    #[test]
    fn process_ontology_level_eliminates_genes_from_ancestors() {
        let (gene_map, ancestors, levels) = two_level_ontology();
        let mut go = GeneOntology::new(gene_map, &ancestors, &levels);
        let target = genes(1..=5);

        let lower: GoElimLevelResults<f64> =
            process_ontology_level(&target, &"2".to_string(), &mut go, 3, 100, 0.05);

        assert_eq!(lower.go_ids, vec!["GO:B"]);
        assert_eq!(lower.hits, vec![5]);
        assert!(lower.pvals[0] < 0.05, "GO:B should be significant");

        // The elimination fired.
        let remaining = go.get_genes(&"GO:A".to_string()).unwrap();
        assert_eq!(remaining.len(), 5);
        assert!(remaining.iter().all(|g| !target.contains(g)));

        let upper: GoElimLevelResults<f64> =
            process_ontology_level(&target, &"1".to_string(), &mut go, 3, 100, 0.05);

        assert_eq!(upper.hits, vec![0]);
        assert_relative_eq!(upper.pvals[0], 1.0, epsilon = 1e-12);
    }

    /// With the threshold at zero nothing clears it, so the ancestor keeps all
    /// ten genes and still sees the full overlap. This is the control for the
    /// test above.
    #[test]
    fn process_ontology_level_without_elimination_leaves_ancestors_intact() {
        let (gene_map, ancestors, levels) = two_level_ontology();
        let mut go = GeneOntology::new(gene_map, &ancestors, &levels);
        let target = genes(1..=5);

        let _: GoElimLevelResults<f64> =
            process_ontology_level(&target, &"2".to_string(), &mut go, 3, 100, 0.0);

        assert_eq!(go.get_genes(&"GO:A".to_string()).unwrap().len(), 10);

        let upper: GoElimLevelResults<f64> =
            process_ontology_level(&target, &"1".to_string(), &mut go, 3, 100, 0.0);

        assert_eq!(upper.hits, vec![5]);
    }

    /// Terms below `min_genes` are dropped before testing, so nothing is
    /// eliminated either.
    #[test]
    fn process_ontology_level_skips_terms_below_min_genes() {
        let (gene_map, ancestors, levels) = two_level_ontology();
        let mut go = GeneOntology::new(gene_map, &ancestors, &levels);

        let res: GoElimLevelResults<f64> =
            process_ontology_level(&genes(1..=5), &"2".to_string(), &mut go, 6, 100, 0.05);

        assert!(res.go_ids.is_empty());
        assert_eq!(go.get_genes(&"GO:A".to_string()).unwrap().len(), 10);
    }

    /// BH runs across every level at once, not per level. For p-values
    /// [0.001, 0.5] and [0.02, 0.9] over n = 4 this gives, in input order,
    /// [0.004, 2/3, 0.04, 0.9]; matches `p.adjust(..., "BH")` in R.
    #[test]
    fn finalise_go_res_applies_bh_across_all_levels() {
        let level_a = GoElimLevelResults::<f64> {
            go_ids: vec!["GO:1".to_string(), "GO:2".to_string()],
            pvals: vec![0.001, 0.5],
            odds_ratios: vec![8.0, 1.0],
            hits: vec![10, 1],
            gene_set_lengths: vec![20, 5],
        };
        let level_b = GoElimLevelResults::<f64> {
            go_ids: vec!["GO:3".to_string(), "GO:4".to_string()],
            pvals: vec![0.02, 0.9],
            odds_ratios: vec![3.0, 1.0],
            hits: vec![5, 2],
            gene_set_lengths: vec![10, 8],
        };

        let all = finalise_go_res(&[level_a.clone(), level_b.clone()], None, None);
        assert_eq!(all.go_ids, vec!["GO:1", "GO:2", "GO:3", "GO:4"]);
        for (g, w) in all.fdr.iter().zip([0.004, 2.0 / 3.0, 0.04, 0.9]) {
            assert_relative_eq!(*g, w, epsilon = 1e-12);
        }

        let filtered = finalise_go_res(&[level_a, level_b], Some(4), Some(0.05));
        assert_eq!(filtered.go_ids, vec!["GO:1", "GO:3"]);
    }
}
