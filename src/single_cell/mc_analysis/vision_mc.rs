//! VISION signature scores for meta cells, see DeTomaso, et al., Nat. Commun.,
//! 2019.
//!
//! The single-cell driver in [`crate::single_cell::sc_analysis::vision`] is
//! generic over [`SingleCellReading`], but VISION is a cell-major algorithm and
//! [`InMemorySparseReader`](crate::single_cell::sc_data::in_memory_io::InMemorySparseReader)
//! only serves the gene-major view. The metacell path therefore walks the CSR
//! rows directly, the way
//! [`calculate_aucell_metacells`](crate::single_cell::mc_analysis::aucell::calculate_aucell_metacells)
//! does, which also keeps the normalised layer at f32 instead of narrowing it
//! to the f16 the disk chunks carry.

use rayon::prelude::*;
use std::borrow::Cow;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_analysis::vision::SignatureGenes;

/////////////
// Helpers //
/////////////

/// Check that every signature gene index addresses a column of the matrix.
///
/// The scoring kernel indexes a dense scratch row directly, so an index past
/// the gene axis would panic rather than score a zero the way the single-cell
/// binary search does. Checking once up front is cheaper than a bounds check
/// per metacell and gives the caller an error instead of an abort.
///
/// This covers the signatures only. The matrix's own `indices`, `indptr` and
/// layer lengths are equally caller-supplied and are checked separately by
/// [`CompressedSparseData2::validate`].
///
/// ### Params
///
/// * `signatures` - The signatures to validate
/// * `n_genes` - Number of genes, i.e. the exclusive bound
///
/// ### Returns
///
/// `Ok(())`, or [`BixverseErrors::SliceIndexOutOfBounds`] for the first
/// offending index.
fn validate_signature_indices(
    signatures: &[SignatureGenes],
    n_genes: usize,
) -> Result<(), BixverseErrors> {
    for sig in signatures {
        for &idx in sig.positive.iter().chain(sig.negative.iter()) {
            if idx >= n_genes {
                return Err(BixverseErrors::SliceIndexOutOfBounds {
                    index: idx,
                    len: n_genes,
                });
            }
        }
    }

    Ok(())
}

/// Score one metacell against every signature.
///
/// `scratch` is a dense `n_genes` buffer owned by the calling thread. The row
/// is scattered into it, every signature gathers by direct index, and only the
/// touched slots are cleared again, so the buffer stays zeroed without a full
/// memset. The trade against the per-gene binary search of the single-cell
/// kernel is `nnz` plus the total signature size here, versus
/// `n_signatures * signature_size * log(nnz)` there, so the dense row wins on
/// the shape metacells have: rows that are dense-ish and a large signature
/// count once the permuted background sets are in play.
///
/// The cell-level mean and variance run over all `n_genes`, including the
/// implicit zeros. Zeros contribute nothing to either sum, so dividing the sums
/// over the stored values by `n_genes` is exactly that. Both accumulate in one
/// pass with f64 accumulators, matching
/// [`crate::single_cell::sc_analysis::vision`]. The f64 is worth more here than
/// it is there: metacell rows are denser and the normalised layer is f32 rather
/// than the f16 the disk chunks carry, so the accumulation error is what sets
/// the floor instead of the storage format.
///
/// ### Params
///
/// * `indices` - Gene indices of the stored values, ascending
/// * `values` - Normalised counts matching `indices`
/// * `signatures` - The signatures to score
/// * `n_genes` - Number of genes, i.e. the length of `scratch`
/// * `scratch` - Thread-local dense row, zeroed on entry and on exit
///
/// ### Returns
///
/// One score per signature, in the order given.
///
/// ### References
///
/// DeTomaso, et al., Nat. Commun., 2019
fn vision_scores_for_metacell(
    indices: &[u32],
    values: &[f32],
    signatures: &[SignatureGenes],
    n_genes: usize,
    scratch: &mut [f32],
) -> Vec<f32> {
    for (&idx, &value) in indices.iter().zip(values.iter()) {
        scratch[idx as usize] = value;
    }

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &value in values {
        let value = value as f64;
        sum += value;
        sum_sq += value * value;
    }

    let mu = sum / n_genes as f64;
    let sigma = ((sum_sq / n_genes as f64) - mu * mu).max(0.0).sqrt();

    let scores: Vec<f32> = signatures
        .iter()
        .map(|sig| {
            let signature_size = (sig.positive.len() + sig.negative.len()) as f64;

            if signature_size == 0.0 || sigma == 0.0 {
                return 0.0;
            }

            let sum_pos: f64 = sig.positive.iter().map(|&idx| scratch[idx] as f64).sum();
            let sum_neg: f64 = sig.negative.iter().map(|&idx| scratch[idx] as f64).sum();

            let mean_sig = (sum_pos - sum_neg) / signature_size;

            // R "znorm_columns" formula
            ((mean_sig - mu) / sigma) as f32
        })
        .collect();

    for &idx in indices {
        scratch[idx as usize] = 0.0;
    }

    scores
}

//////////
// Main //
//////////

/// Calculate VISION signature scores across a set of meta cells
///
/// In-memory version of VISION that operates directly on
/// [`CompressedSparseData2`]. Reads normalised counts from `data_2`. Every row
/// is scored; subset the matrix upstream if only part of it is of interest.
///
/// ### Params
///
/// * `matrix` - Sparse matrix in CSR or CSC format, shape (metacells, genes).
///   `data_2` must contain the normalised counts.
/// * `gene_signs` - Slice of [`SignatureGenes`] to calculate the scores for.
///   All gene indices must address a column of `matrix`.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// A `Vec<Vec<f32>>` with metacells x scores per gene set (pair), the same
/// layout the single-cell driver returns, so it feeds
/// [`calc_autocorr_with_clusters`](crate::single_cell::sc_analysis::vision::calc_autocorr_with_clusters)
/// as is.
///
/// ### References
///
/// DeTomaso, et al., Nat. Commun., 2019
pub fn calculate_vision_metacells<T>(
    matrix: &CompressedSparseData2<T, f32>,
    gene_signs: &[SignatureGenes],
    verbose: usize,
) -> Result<Vec<Vec<f32>>, BixverseErrors>
where
    T: BixverseNumeric,
{
    let verbosity = parse_verbosity_level(verbose);

    matrix.validate()?;

    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csc => Cow::Owned(matrix.transform()),
    };

    let (n_metacells, n_genes) = csr.shape;
    let data_2 = csr
        .data_2
        .as_ref()
        .ok_or(BixverseErrors::Data2NotAvailable)?;

    if data_2.len() < csr.indices.len() {
        return Err(BixverseErrors::DimensionMisMatchSparse {
            indices_len: csr.indices.len(),
            data_len: data_2.len(),
        });
    }

    validate_signature_indices(gene_signs, n_genes)?;

    let start_signatures = Instant::now();
    let signature_scores: Vec<Vec<f32>> = (0..n_metacells)
        .into_par_iter()
        .map_init(
            || vec![0.0_f32; n_genes],
            |scratch, row| {
                let lo = csr.indptr[row] as usize;
                let hi = csr.indptr[row + 1] as usize;

                vision_scores_for_metacell(
                    &csr.indices[lo..hi],
                    &data_2[lo..hi],
                    gene_signs,
                    n_genes,
                    scratch,
                )
            },
        )
        .collect();

    if verbosity.normal_verbosity() {
        println!(
            "Calculated VISION scores: {:.2?}",
            start_signatures.elapsed()
        );
    }

    // metacells x signatures
    Ok(signature_scores)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let n_rows = dense.len();
        let n_cols = dense[0].len();

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
            shape: (n_rows, n_cols),
        }
    }

    /// Reference implementation of the kernel, straight off the formula.
    ///
    /// ### Params
    ///
    /// * `row` - The dense row
    /// * `sig` - The signature to score
    ///
    /// ### Returns
    ///
    /// The VISION score.
    fn reference_score(row: &[f32], sig: &SignatureGenes) -> f32 {
        let n = row.len() as f64;
        let sum: f64 = row.iter().map(|&v| v as f64).sum();
        let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();

        let mu = sum / n;
        let sigma = ((sum_sq / n) - mu * mu).max(0.0).sqrt();

        let size = (sig.positive.len() + sig.negative.len()) as f64;
        if size == 0.0 || sigma == 0.0 {
            return 0.0;
        }

        let sum_pos: f64 = sig.positive.iter().map(|&i| row[i] as f64).sum();
        let sum_neg: f64 = sig.negative.iter().map(|&i| row[i] as f64).sum();

        (((sum_pos - sum_neg) / size - mu) / sigma) as f32
    }

    fn toy_dense() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 3.0, 0.5, 0.0, 2.0],
            vec![0.0, 4.0, 0.0, 0.0, 1.5, 0.25],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    }

    #[test]
    fn test_vision_metacells_matches_dense_reference() {
        let dense = toy_dense();
        let matrix = csr_from_dense(&dense);

        let signatures = vec![
            SignatureGenes {
                positive: vec![0, 2, 5],
                negative: vec![],
            },
            SignatureGenes {
                positive: vec![1, 4],
                negative: vec![3],
            },
        ];

        let scores = calculate_vision_metacells(&matrix, &signatures, 0).unwrap();

        assert_eq!(scores.len(), dense.len());
        for (row, row_scores) in dense.iter().zip(scores.iter()) {
            assert_eq!(row_scores.len(), signatures.len());
            for (sig, &got) in signatures.iter().zip(row_scores.iter()) {
                assert_relative_eq!(got, reference_score(row, sig), epsilon = 1e-6);
            }
        }
    }

    /// The CSC arm goes through `transform()`, which has to land on the same
    /// numbers as the borrowed CSR one.
    #[test]
    fn test_vision_metacells_csc_and_csr_agree() {
        let dense = toy_dense();
        let csr = csr_from_dense(&dense);
        let csc = csr.transform();

        let signatures = vec![SignatureGenes {
            positive: vec![0, 2, 5],
            negative: vec![1],
        }];

        let from_csr = calculate_vision_metacells(&csr, &signatures, 0).unwrap();
        let from_csc = calculate_vision_metacells(&csc, &signatures, 0).unwrap();

        assert_eq!(from_csr.len(), from_csc.len());
        for (a, b) in from_csr.iter().zip(from_csc.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_relative_eq!(x, y, epsilon = 1e-6);
            }
        }
    }

    /// The dense scratch is reused across rows, so a row must not see the
    /// values of the one scored before it on the same thread.
    #[test]
    fn test_vision_metacells_scratch_is_left_clean() {
        let dense = vec![
            vec![5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        ];

        let signatures = vec![SignatureGenes {
            positive: vec![0, 1, 2],
            negative: vec![],
        }];

        let mut scratch = vec![0.0_f32; 6];
        let matrix = csr_from_dense(&dense);
        let data_2 = matrix.data_2.as_ref().unwrap();

        let mut scores = Vec::new();
        for row in 0..dense.len() {
            let lo = matrix.indptr[row] as usize;
            let hi = matrix.indptr[row + 1] as usize;
            scores.push(vision_scores_for_metacell(
                &matrix.indices[lo..hi],
                &data_2[lo..hi],
                &signatures,
                6,
                &mut scratch,
            ));
        }

        assert!(scratch.iter().all(|&v| v == 0.0));
        // the second row has no expression on the signature genes, so it has
        // to score the negative of its own mean over sigma
        assert_relative_eq!(scores[1][0], reference_score(&dense[1], &signatures[0]));
    }

    #[test]
    fn test_vision_metacells_negative_genes_subtract() {
        let dense = vec![vec![1.0, 0.0, 3.0, 0.5, 0.0, 2.0]];
        let matrix = csr_from_dense(&dense);

        let positive_only = vec![SignatureGenes {
            positive: vec![0, 2],
            negative: vec![],
        }];
        let with_negative = vec![SignatureGenes {
            positive: vec![0, 2],
            negative: vec![5],
        }];

        let pos = calculate_vision_metacells(&matrix, &positive_only, 0).unwrap()[0][0];
        let mixed = calculate_vision_metacells(&matrix, &with_negative, 0).unwrap()[0][0];

        assert!(mixed < pos);
        assert_relative_eq!(mixed, reference_score(&dense[0], &with_negative[0]));
    }

    #[test]
    fn test_vision_metacells_degenerate_signatures() {
        // second row is constant, so sigma is zero
        let dense = vec![vec![1.0, 0.0, 3.0], vec![2.0, 2.0, 2.0]];
        let matrix = csr_from_dense(&dense);

        let signatures = vec![
            SignatureGenes {
                positive: vec![],
                negative: vec![],
            },
            SignatureGenes {
                positive: vec![0, 2],
                negative: vec![],
            },
        ];

        let scores = calculate_vision_metacells(&matrix, &signatures, 0).unwrap();

        assert_eq!(scores[0][0], 0.0);
        assert_eq!(scores[1][0], 0.0);
        assert_eq!(scores[1][1], 0.0);
    }

    #[test]
    fn test_vision_metacells_rejects_out_of_range_gene() {
        let matrix = csr_from_dense(&toy_dense());

        let signatures = vec![SignatureGenes {
            positive: vec![0],
            negative: vec![6],
        }];

        let res = calculate_vision_metacells(&matrix, &signatures, 0);

        assert!(matches!(
            res,
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 6, len: 6 })
        ));
    }

    /// The matrix comes from R with public fields and no enforced invariant, so
    /// a bad gene index has to error rather than panic inside a rayon worker
    /// when the scatter writes into the dense scratch.
    #[test]
    fn test_vision_metacells_rejects_a_gene_index_past_the_axis() {
        let mut matrix = csr_from_dense(&toy_dense());
        let last = matrix.indices.len() - 1;
        matrix.indices[last] = 99;

        let signatures = vec![SignatureGenes {
            positive: vec![0],
            negative: vec![],
        }];

        assert!(matches!(
            calculate_vision_metacells(&matrix, &signatures, 0),
            Err(BixverseErrors::SliceIndexOutOfBounds { index: 99, len: 6 })
        ));
    }

    /// An over-long `indptr` used to silently return fewer score rows than the
    /// matrix declares metacells, which is worse than a panic.
    #[test]
    fn test_vision_metacells_rejects_an_inconsistent_indptr() {
        let mut matrix = csr_from_dense(&toy_dense());
        matrix.indptr.push(*matrix.indptr.last().unwrap());

        let signatures = vec![SignatureGenes {
            positive: vec![0],
            negative: vec![],
        }];

        assert!(matches!(
            calculate_vision_metacells(&matrix, &signatures, 0),
            Err(BixverseErrors::SparseIndptrInvalid { .. })
        ));
    }

    #[test]
    fn test_vision_metacells_requires_data_2() {
        let mut matrix = csr_from_dense(&toy_dense());
        matrix.data_2 = None;

        let signatures = vec![SignatureGenes {
            positive: vec![0],
            negative: vec![],
        }];

        let res = calculate_vision_metacells(&matrix, &signatures, 0);

        assert!(matches!(res, Err(BixverseErrors::Data2NotAvailable)));
    }
}
