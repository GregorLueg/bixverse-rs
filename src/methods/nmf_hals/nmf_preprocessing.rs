//! Pre-processing methods for NMF on sparse and dense data.

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::core::math::matrix_helpers::col_sds;
use crate::core::math::sparse::sparse_col_sds_csc;
use crate::prelude::*;

//////////
// Enum //
//////////

use crate::prelude::BixverseFloat;

/// Options to do NmfPreprocessing
#[derive(Clone, Copy, Debug, Default)]
pub enum NmfPreprocessing {
    /// No pre-processing.
    #[default]
    None,
    /// Scale each column (features) by the standard deviation.
    SdScaling,
    /// Scale each column (features) by the squared standard deviation.
    SqrtSdScaling,
}

/// Transpose sparse counts to CSC if they arrived as CSR.
///
/// [`nmf_process_sparse`] indexes `indptr` per column, so CSR input scales the
/// wrong axis and returns quiet nonsense rather than an error. The R side always
/// hands over CSR, so this is the common path, not the exception.
///
/// ### Params
///
/// * `data` - The counts, in either orientation.
/// * `verbosity` - Resolved verbosity, used only for the transpose notice.
///
/// ### Returns
///
/// The same data in CSC orientation.
pub fn ensure_csc(
    data: CompressedSparseData2<f32>,
    verbosity: Verbosity,
) -> CompressedSparseData2<f32> {
    if data.cs_type.is_csr() {
        if verbosity.detailed_verbosity() {
            println!("NMF: data was provided as CSR. Transposing to CSC.")
        }
        data.transform()
    } else {
        data
    }
}

/// Parse the NMF pre-processing
///
/// ### Params
///
/// * `s` - String to parse. `"none"` -> leave data as is; `"sd" | "scaling"`
///   -> divide the respective features by their standard deviation; `"sqrt_sd"`
///   -> divide the respective features by the square root of their standard
///   deviation.
///
/// ### Returns
///
/// The option of [NmfPreprocessing]
pub fn parse_nmf_processing(s: &str) -> Option<NmfPreprocessing> {
    match s.to_lowercase().as_str() {
        "none" => Some(NmfPreprocessing::None),
        "sd" | "scaling" => Some(NmfPreprocessing::SdScaling),
        "sqrt_sd" => Some(NmfPreprocessing::SqrtSdScaling),
        _ => None,
    }
}

///////////
// Dense //
///////////

/// Process dense matrices of samples x features
///
/// ### Params
///
/// * `x` - Matrix to preprocess
/// * `process` - Which approach of NMF pre-processing to apply, see
///   [NmfPreprocessing]
///
/// ### Returns
///
/// The pre-processed matrix
pub fn nmf_process_dense<T: BixverseFloat>(x: MatRef<T>, process: &NmfPreprocessing) -> Mat<T> {
    let epsilon = T::from_f32(1e-6).unwrap();

    match process {
        NmfPreprocessing::None => x.to_owned(),
        NmfPreprocessing::SdScaling => {
            let mut sds = col_sds(x);
            // safe guard against tiny values...
            sds.par_iter_mut()
                .for_each(|x| *x = if *x <= epsilon { T::one() } else { *x });

            Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] / sds[j])
        }
        NmfPreprocessing::SqrtSdScaling => {
            let mut sds = col_sds(x);
            sds.par_iter_mut()
                .for_each(|x| *x = if *x <= epsilon { T::one() } else { x.sqrt() });

            Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] / sds[j])
        }
    }
}

////////////
// Sparse //
////////////

/// Process sparse matrices of samples x features (need to be in CSC format!)
///
/// ### Params
///
/// * `x` - Sparse matrix to preprocess, see [CompressedSparseData2].
/// * `process` - Which approach of NMF pre-processing to apply, see
///   [NmfPreprocessing].
/// * `use_second_layer` - Use the second data layer.
///
/// ### Returns
///
/// The pre-processed matrix
pub fn nmf_process_sparse<T>(
    x: &CompressedSparseData2<T>,
    process: &NmfPreprocessing,
    use_second_layer: bool,
) -> Result<CompressedSparseData2<T>, BixverseErrors>
where
    T: BixverseFloat + BixverseSimd,
{
    let epsilon = T::from_f32(1e-6).unwrap();

    match process {
        NmfPreprocessing::None => Ok(x.clone()),
        NmfPreprocessing::SdScaling | NmfPreprocessing::SqrtSdScaling => {
            let use_sqrt = matches!(process, NmfPreprocessing::SqrtSdScaling);
            let mut sds = sparse_col_sds_csc(x, use_second_layer)?;

            sds.par_iter_mut().for_each(|s| {
                *s = if *s <= epsilon {
                    T::one()
                } else if use_sqrt {
                    s.sqrt()
                } else {
                    *s
                };
            });

            let mut out = x.clone();
            let active_data: &mut Vec<T> = if use_second_layer {
                out.data_2
                    .as_mut()
                    .ok_or(BixverseErrors::Data2NotAvailable)?
            } else {
                &mut out.data
            };

            let ncols = x.shape().1;
            for j in 0..ncols {
                let start = x.indptr[j] as usize;
                let end = x.indptr[j + 1] as usize;
                let divisor = sds[j];
                for v in &mut active_data[start..end] {
                    *v /= divisor;
                }
            }

            Ok(out)
        }
    }
}
