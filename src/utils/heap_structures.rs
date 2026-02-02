use std::cmp::Ordering;

use crate::prelude::*;

/// Wrapper for generic float to use in BinaryHeap (min-heap)
#[derive(Clone, Copy)]
pub struct OrderedFloat<T: BixverseFloat>(pub T);

impl<T> OrderedFloat<T>
where
    T: BixverseFloat,
{
    pub fn get_value(&self) -> T {
        self.0
    }
}

impl<T: BixverseFloat> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl<T: BixverseFloat> Eq for OrderedFloat<T> {}

impl<T: BixverseFloat> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: BixverseFloat> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}
