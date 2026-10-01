//! Dense GEMM entry points. `gemm` is the `ann-search-rs` drop-in for faer's
//! `matmul` that routes through Apple Accelerate under the `accelerate`
//! feature on macOS. `gram` covers the symmetric products that would otherwise
//! go through faer's triangular matmul.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Accum, MatMut, MatRef, Par};
use faer_traits::ComplexField;

pub use ann_search_rs::utils::gemm::gemm;

/// Symmetric product such as `X^T X` or `X X^T`, written into one triangle
///
/// With Accelerate the full square is computed: it still beats faer's
/// triangle-only kernel despite twice the flops. Callers must therefore only
/// read the triangle named by `structure`; the other one may be overwritten
/// (or accumulated into under `Accum::Add`).
///
/// ### Params
///
/// * `dst` - Square destination
/// * `structure` - Triangle of `dst` the caller reads
/// * `accum` - Replace or accumulate into `dst`
/// * `lhs` - Left operand
/// * `rhs` - Right operand
/// * `alpha` - Scale of the product
/// * `par` - Parallelism for the faer path
#[inline]
pub fn gram<T: ComplexField>(
    dst: MatMut<T>,
    structure: BlockStructure,
    accum: Accum,
    lhs: MatRef<T>,
    rhs: MatRef<T>,
    alpha: T,
    par: Par,
) {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        let _ = structure;
        gemm(dst, accum, lhs, rhs, alpha, par);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    faer::linalg::matmul::triangular::matmul(
        dst,
        structure,
        accum,
        lhs,
        BlockStructure::Rectangular,
        rhs,
        BlockStructure::Rectangular,
        alpha,
        par,
    );
}
