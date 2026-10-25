//! decoupler-style ULM and MLM activity scores for meta cells, see
//! Badia-i-Mompel, et al., Bioinform. Adv., 2022.
//!
//! Twin of [`crate::sc_analysis::lm_activity`] that walks the CSR rows of an
//! in-memory metacell matrix directly, since
//! [`InMemorySparseReader`](bixverse_sc_data::in_memory_io::InMemorySparseReader)
//! only serves the gene-major view. This also keeps the normalised layer at
//! f32 instead of the f16 the disk chunks carry.

use faer::Mat;
use std::borrow::Cow;
use std::time::Instant;

use crate::prelude::*;
use bixverse_methods::enrichment::lm_activity::{
    FeatureSetWeights, LmActivityParams, LmActivityRes, LmScorer, ObsCrossProducts,
};

/// ULM or MLM activity scores across a set of meta cells
///
/// Every metacell is one fit over all genes of the matrix, the implicit zeros
/// included, on the normalised counts in `data_2`. Every row is scored in one
/// batch; subset the matrix upstream if only part of it is of interest.
///
/// ### Params
///
/// * `matrix` - Sparse matrix in CSR or CSC format, shape (metacells, genes).
///   `data_2` must contain the normalised counts.
/// * `net` - The feature sets over the genes of the matrix.
/// * `params` - Method, statistic and p-value switch. `None` gives ULM t
///   values without p-values.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The scores, metacells x sets.
///
/// ### References
///
/// Badia-i-Mompel, et al., Bioinform. Adv., 2022
pub fn calculate_lm_activity_metacells<T>(
    matrix: &CompressedSparseData2<T, f32>,
    net: &FeatureSetWeights,
    params: Option<LmActivityParams>,
    verbose: usize,
) -> Result<LmActivityRes<f32>, ScErrors>
where
    T: BixverseNumeric,
{
    let verbosity = parse_verbosity_level(verbose);
    let params = params.unwrap_or_default();

    matrix.validate()?;

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csc => Cow::Owned(matrix.transform()),
    };

    let (n_metacells, n_genes) = csr.shape;
    let data_2 = csr.data_2.as_ref().ok_or(CoreErrors::Data2NotAvailable)?;

    if data_2.len() < csr.indices.len() {
        return Err(CoreErrors::DimensionMisMatchSparse {
            indices_len: csr.indices.len(),
            data_len: data_2.len(),
        }
        .into());
    }

    // with validate() this bounds every index cross_products sees
    if n_genes != net.n_features {
        return Err(MethodsErrors::LmActivityShapeMismatch {
            expected: net.n_features,
            got: n_genes,
        }
        .into());
    }
    let scorer = LmScorer::new(net, &params)?;

    let start = Instant::now();
    let cp = ObsCrossProducts::new(net, n_metacells, |o| {
        let lo = csr.indptr[o] as usize;
        let hi = csr.indptr[o + 1] as usize;
        csr.indices[lo..hi]
            .iter()
            .zip(&data_2[lo..hi])
            .map(|(&i, &v)| (i as usize, v as f64))
    });
    let (es, pv) = scorer.score(&cp)?;

    if verbosity.normal_verbosity() {
        println!(
            "Calculated linear model activity scores: {:.2?}",
            start.elapsed()
        );
    }

    let n_sets = net.n_sets;
    let to_mat = |v: Vec<f64>| Mat::from_fn(n_metacells, n_sets, |i, j| v[i * n_sets + j] as f32);
    Ok(LmActivityRes {
        es: to_mat(es),
        p_values: pv.map(to_mat),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use bixverse_methods::enrichment::lm_activity::{LmMethod, LmStat, lm_activity};

    /// Toy metacells x genes matrix, dense row-major.
    ///
    /// ### Returns
    ///
    /// The rows, roughly a third of the entries non-zero.
    fn toy_dense() -> Vec<Vec<f32>> {
        (0..10)
            .map(|c| {
                (0..40)
                    .map(|g| {
                        if (g * 5 + c * 3) % 3 == 0 {
                            ((g + 2 * c) % 7) as f32 * 0.37 + 0.1
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Build a CSR matrix from a dense row-major representation.
    ///
    /// ### Params
    ///
    /// * `dense` - The rows, all of the same length
    ///
    /// ### Returns
    ///
    /// The CSR matrix with the values in both layers.
    fn csr_from_dense(dense: &[Vec<f32>]) -> CompressedSparseData2<f32, f32> {
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut indptr: Vec<u32> = vec![0];
        for row in dense {
            for (col, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    data.push(value);
                    indices.push(col as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        CompressedSparseData2 {
            data: data.clone(),
            indices,
            indptr,
            cs_type: CompressedSparseFormat::Csr,
            data_2: Some(data),
            shape: (dense.len(), dense[0].len()),
        }
    }

    /// Weighted feature sets over the 40 toy genes.
    ///
    /// ### Returns
    ///
    /// The network.
    fn toy_net() -> FeatureSetWeights {
        let sets = vec![
            vec![0, 4, 8, 12, 16],
            vec![1, 4, 9, 30, 31, 35],
            vec![2, 3, 20, 21],
        ];
        let weights = vec![
            vec![1.0, 1.0, -1.0, 0.5, 2.0],
            vec![1.0, -0.5, 1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        FeatureSetWeights::new(&sets, Some(&weights), 40).unwrap()
    }

    #[test]
    fn test_lm_activity_metacells_matches_dense() {
        let dense = toy_dense();
        let matrix = csr_from_dense(&dense);
        let net = toy_net();
        let mat = Mat::<f64>::from_fn(dense[0].len(), dense.len(), |g, c| dense[c][g] as f64);

        for method in [LmMethod::Ulm, LmMethod::Mlm] {
            let params = LmActivityParams::new(method, LmStat::TValue, true);
            let mc = calculate_lm_activity_metacells(&matrix, &net, Some(params), 0).unwrap();
            let reference = lm_activity(mat.as_ref(), &net, Some(params)).unwrap();
            let (mc_p, ref_p) = (mc.p_values.unwrap(), reference.p_values.unwrap());
            for c in 0..dense.len() {
                for s in 0..net.n_sets {
                    assert_relative_eq!(
                        mc.es[(c, s)] as f64,
                        reference.es[(c, s)],
                        max_relative = 1e-6
                    );
                    assert_relative_eq!(mc_p[(c, s)] as f64, ref_p[(c, s)], max_relative = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_lm_activity_metacells_csc_matches_csr() {
        let csr = csr_from_dense(&toy_dense());
        let csc = csr.transform();
        let net = toy_net();
        let params = LmActivityParams::new(LmMethod::Mlm, LmStat::TValue, true);

        let a = calculate_lm_activity_metacells(&csr, &net, Some(params), 0).unwrap();
        let b = calculate_lm_activity_metacells(&csc, &net, Some(params), 0).unwrap();
        assert_eq!(a.es, b.es);
        assert_eq!(a.p_values, b.p_values);
    }

    #[test]
    fn test_lm_activity_metacells_rejects_gene_mismatch() {
        let matrix = csr_from_dense(&toy_dense());
        let net = FeatureSetWeights::new::<f64>(&[vec![0, 1, 2]], None, 41).unwrap();
        assert!(matches!(
            calculate_lm_activity_metacells(&matrix, &net, None, 0),
            Err(ScErrors::Methods(
                MethodsErrors::LmActivityShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn test_lm_activity_metacells_requires_data_2() {
        let mut matrix = csr_from_dense(&toy_dense());
        matrix.data_2 = None;
        assert!(matches!(
            calculate_lm_activity_metacells(&matrix, &toy_net(), None, 0),
            Err(ScErrors::Core(CoreErrors::Data2NotAvailable))
        ));
    }
}
