//! Various functions, structures, etc. to expose more broadly when using this
//! crate in other libraries

use std::time::Duration;
use thousands::Separable;

pub use crate::core::math::MAD_SCALE;
pub use crate::core::math::sparse::{
    CompressedSparseData2, CompressedSparseFormat, LanczosParams, SparseAxis, SparseFromDense,
    parse_compressed_sparse_format,
};
pub use crate::errors::*;
pub use crate::utils::heap_structures::RevOrderedFloat;
pub use crate::utils::matrix_utils::*;
pub use crate::utils::simd::BixverseSimd;
pub use crate::utils::traits::*;
pub use crate::utils::vec_utils::*;
pub use crate::{assert_nrows, assert_same_len, assert_symmetric_mat};

////////////
// Consts //
////////////

/// Step size, in percent, between progress reports.
///
/// Reporting on decile crossings rather than per unit of work bounds the number
/// of lines a sweep can emit at ten, no matter whether it walks ten blocks or
/// ten thousand genes.
const PROGRESS_STEP_PCT: usize = 10;

pub use ann_search_rs::{Verbosity, parse_annsearch_verbosity as parse_verbosity_level};

/// Map an on/off verbose flag onto [`Verbosity`].
///
/// For callers that still carry `verbose: bool` and hand it to ann-search-rs.
///
/// ### Params
///
/// * `verbose` - The flag
///
/// ### Returns
///
/// [`Verbosity::Normal`] if set, [`Verbosity::Quiet`] otherwise.
pub fn verbosity_from_flag(verbose: bool) -> Verbosity {
    if verbose {
        Verbosity::Normal
    } else {
        Verbosity::Quiet
    }
}

//////////////
// Progress //
//////////////

/// Prints a progress line whenever a sweep crosses a decile of its work.
///
/// Cheap enough to call on every unit of work: it formats nothing unless `done`
/// and `prev_done` fall either side of a [PROGRESS_STEP_PCT] boundary, or the
/// sweep has just finished. The verbosity check stays with the caller, since
/// not every sweep reports its progress at the same level.
///
/// ### Params
///
/// * `done` - Units of work finished, including the one just completed
/// * `prev_done` - Units of work finished before it
/// * `total` - Units of work in the whole sweep. Nothing prints if this is `0`
/// * `unit` - What is being counted, e.g. `"genes"`. Printed as given
/// * `elapsed` - Time since the sweep started
pub fn report_decile_progress(
    done: usize,
    prev_done: usize,
    total: usize,
    unit: &str,
    elapsed: Duration,
) {
    if total == 0 {
        return;
    }

    let pct = done * 100 / total;
    let prev_pct = prev_done * 100 / total;

    if pct / PROGRESS_STEP_PCT > prev_pct / PROGRESS_STEP_PCT || done == total {
        println!(
            "  Progress: {pct}% ({} / {} {unit}, {elapsed:.2?})",
            done.separate_with_underscores(),
            total.separate_with_underscores()
        );
    }
}
