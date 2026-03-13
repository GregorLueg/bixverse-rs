//! All type of utility modules with shared code that does not fit fully into
//! other modules. Has assertion macros, structures designed for the heap,
//! traits and their implementations, R <> Rust interface functions and more.

pub mod heap_structures;
pub mod macros;
pub mod matrix_utils;
pub mod r_rust_interface;
pub mod simd;
pub mod traits;
pub mod vec_utils;

use rustc_hash::{FxBuildHasher, FxHashSet};

///////////////////
// General utils //
///////////////////

/// String slice to FxHashSet
///
/// ### Params
///
/// * `x` - The string slice.
///
/// ### Returns
///
/// A HashSet with borrowed String values
pub fn string_vec_to_set(x: &[String]) -> FxHashSet<&String> {
    let mut set = FxHashSet::with_capacity_and_hasher(x.len(), FxBuildHasher);
    for s in x {
        set.insert(s);
    }
    set
}
