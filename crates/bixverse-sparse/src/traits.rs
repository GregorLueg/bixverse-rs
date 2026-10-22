//! Numeric trait bounds shared by the sparse structures.

use std::ops::{Add, AddAssign, DivAssign, Mul, Sub};

/// More general numeric trait that includes also integers
pub trait BixverseNumeric:
    Clone
    + Default
    + Copy
    + Sync
    + Send
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + PartialEq
    + Sub
    + DivAssign
{
}

impl<T> BixverseNumeric for T where
    T: Clone
        + Default
        + Copy
        + Sync
        + Send
        + Add<Output = Self>
        + Mul<Output = Self>
        + Sub<Output = Self>
        + AddAssign
        + PartialEq
        + Sub
        + DivAssign
{
}

///////////////////
// Index casting //
///////////////////

/// Casts index vectors between `usize` and `u32` (memory-width conversion).
pub trait VecIndexCast<U> {
    /// Casts the index vector element-wise to the target integer type.
    fn index_cast(self) -> Vec<U>;
}

impl VecIndexCast<u32> for Vec<usize> {
    fn index_cast(self) -> Vec<u32> {
        self.into_iter().map(|x| x as u32).collect()
    }
}

impl VecIndexCast<u32> for &[usize] {
    fn index_cast(self) -> Vec<u32> {
        self.iter().map(|&x| x as u32).collect()
    }
}

impl VecIndexCast<usize> for Vec<u32> {
    fn index_cast(self) -> Vec<usize> {
        self.into_iter().map(|x| x as usize).collect()
    }
}

impl VecIndexCast<usize> for &[u32] {
    fn index_cast(self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
}
