//! Contains structures designed to be put on the heap. One key application is
//! checking distances via the [RevOrderedFloat].

use std::cmp::Ordering;

use crate::prelude::*;

/// Wrapper for generic float to use in BinaryHeap (min-heap)
#[derive(Clone, Copy)]
pub struct RevOrderedFloat<T: BixverseFloat>(pub T);

impl<T> RevOrderedFloat<T>
where
    T: BixverseFloat,
{
    /// Returns the value stored in the RevOrderedFloat
    pub fn get_value(&self) -> T {
        self.0
    }
}

impl<T: BixverseFloat> PartialEq for RevOrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl<T: BixverseFloat> Eq for RevOrderedFloat<T> {}

impl<T: BixverseFloat> PartialOrd for RevOrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: BixverseFloat> Ord for RevOrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}
