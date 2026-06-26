//! Shared types and helpers for spatially variable gene detection.

use crate::single_cell::sc_data::data_io::CscGeneChunk;

////////////
// Params //
////////////

/// Which expression layer to read from a [`CscGeneChunk`].
///
/// `CscGeneChunk` carries both raw counts (`data_raw`) and normalised counts
/// (`data_norm`). This enum picks one. Moran's I supports either; SPARK-X is
/// forced to raw (scale-invariant per gene).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assay {
    /// Use raw counts from `CscGeneChunk::data_raw`.
    Raw,
    /// Use normalised counts from `CscGeneChunk::data_norm`.
    Norm,
}

/// Common parameters for SVG detection methods.
#[derive(Debug, Clone)]
pub struct SpatialSvgParams {
    /// Which expression layer to use. Default: normalised.
    pub assay: Assay,
}

impl Default for SpatialSvgParams {
    fn default() -> Self {
        Self {
            assay: Assay::Norm,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Materialise a `CscGeneChunk` into a dense per-spot `f32` vector.
///
/// **Precondition:** the chunk must have already been passed through
/// `CscGeneChunk::filter_selected_cells` with the per-sample cell set, so that
/// `chunk.indices` are local positions in `0..n_spots` (the standard
/// HotSpot/MoransI per-sample pipeline). `out` is zeroed and the non-zero
/// entries are scattered into the matching local position.
///
/// ### Params
///
/// * `chunk` - The filtered CSC gene chunk for a single gene
/// * `assay` - Which expression layer to read
/// * `out` - Dense output vector, length `n_spots`. Zeroed by this function.
pub fn materialise_gene_dense(chunk: &CscGeneChunk, assay: Assay, out: &mut [f32]) {
    out.fill(0.0);

    match assay {
        Assay::Raw => {
            for (local, raw_val) in chunk.indices.iter().zip(chunk.data_raw.iter()) {
                out[*local as usize] = raw_val as f32;
            }
        }
        Assay::Norm => {
            for (local, norm_val) in chunk.indices.iter().zip(chunk.data_norm.iter()) {
                out[*local as usize] = norm_val.to_f32();
            }
        }
    }
}

/// Centre values in place: subtract the mean, return `(mean, sum_sq)` where
/// `sum_sq = sum_i (x_i - mean)^2` after centring.
///
/// Used by Moran's I: the centred vector is the input to the quadratic form,
/// and `sum_sq` is the denominator of the test statistic.
pub fn center_inplace(vals: &mut [f32]) -> (f32, f32) {
    let n = vals.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let inv_n = 1.0_f32 / n as f32;

    let mut sum = 0.0_f32;
    for &v in vals.iter() {
        sum += v;
    }
    let mean = sum * inv_n;

    let mut sum_sq = 0.0_f32;
    for v in vals.iter_mut() {
        *v -= mean;
        sum_sq += *v * *v;
    }

    (mean, sum_sq)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_inplace_zeros_mean() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let (mean, ss) = center_inplace(&mut v);
        assert!((mean - 3.0).abs() < 1e-6);
        // centred: [-2, -1, 0, 1, 2], ss = 4+1+0+1+4 = 10
        assert!((ss - 10.0).abs() < 1e-6);
        let new_mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(new_mean.abs() < 1e-6);
    }

    #[test]
    fn center_inplace_constant_is_zero() {
        let mut v = vec![3.0_f32; 10];
        let (mean, ss) = center_inplace(&mut v);
        assert!((mean - 3.0).abs() < 1e-6);
        assert_eq!(ss, 0.0);
    }

    #[test]
    fn center_inplace_empty_is_noop() {
        let mut v: Vec<f32> = Vec::new();
        let (mean, ss) = center_inplace(&mut v);
        assert_eq!(mean, 0.0);
        assert_eq!(ss, 0.0);
    }
}
