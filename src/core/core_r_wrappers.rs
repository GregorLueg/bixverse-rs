//! Specific R wrappers over structures or functions within the `core` module

use extendr_api::prelude::*;
use std::collections::BTreeMap;

use crate::core::mat_struct::NamedMatrix;
use crate::prelude::*;

impl<'a, T> NamedMatrix<'a, T>
where
    T: BixverseFloat,
{
    /// Generate a NamedMatrix from an RMatrix.
    pub fn from_r_matrix(x: &'a RMatrix<f64>) -> NamedMatrix<'a, f64> {
        let col_names: BTreeMap<String, usize> = x
            .get_colnames()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect();
        let row_names: BTreeMap<String, usize> = x
            .get_rownames()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect();
        let mat = r_matrix_to_faer(x);
        NamedMatrix {
            col_names,
            row_names,
            values: mat,
        }
    }
}
