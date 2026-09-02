//! Meta cells analysis methods. These are methods that leverage the aggregated
//! counts from the meta-cells. Focus on 'bag-of-genes' methods, such as
//! SCENIC, etc.

pub mod aucell;
pub mod dialogue_mc;
pub mod hotspot_mc;
pub mod metacell_density;
pub mod metrics;
pub mod nebula_mc;
pub mod nmf_mc;
pub mod scenic_metacells;
pub mod vision_mc;

use std::borrow::Cow;

use crate::prelude::*;

/// Coerces a metacell matrix to the gene-major layout the in-memory reader
/// needs.
///
/// Borrows when it already is CSC, so the common case costs nothing.
///
/// ### Params
///
/// * `matrix` - The metacell counts in either orientation
///
/// ### Returns
///
/// The matrix as CSC.
pub(crate) fn as_csc(
    matrix: &CompressedSparseData2<u32, f32>,
) -> Cow<'_, CompressedSparseData2<u32, f32>> {
    match matrix.cs_type {
        CompressedSparseFormat::Csc => Cow::Borrowed(matrix),
        CompressedSparseFormat::Csr => Cow::Owned(matrix.transform()),
    }
}
