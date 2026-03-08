//! Various traits in bixverse

use extendr_api::*;
use faer::traits::{ComplexField, RealField};
use faer_entity::SimpleEntity;
use num_traits::float::TotalOrder;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Sub, SubAssign};

///////////////////
// Numeric trait //
///////////////////

/// Trait for floating-point types used in Bixverse. Has all of the common
/// floating-point operations and traits.
pub trait BixverseFloat:
    Float
    + FromPrimitive
    + ToPrimitive
    + ComplexField
    + Copy
    + 'static
    + RealField
    + AddAssign
    + DivAssign
    + SubAssign
    + Div
    + Display
    + Sub
    + Mul
    + Div
    + TotalOrder
    + Clone
{
}

impl<T> BixverseFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + ComplexField
        + Copy
        + 'static
        + RealField
        + AddAssign
        + DivAssign
        + SubAssign
        + Div
        + Display
        + Sub
        + Mul
        + Div
        + TotalOrder
        + Clone
{
}

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

////////////////////
// R vector stuff //
////////////////////

/// Converts vectors between integer types for R compatibility.
pub trait VecConvert<U> {
    /// Converts usize vectors into i32 for fast R transfer
    fn r_int_convert(self) -> Vec<U>;

    /// Converts vectors from usize to i32, while adding one (shifting from
    /// 0-indexed data to 1-indexed data) or vice versa from i32 to usize while
    /// subtracting 1 (shifting from 1-indexed data to 0-indexed data).
    fn r_int_convert_shift(self) -> Vec<U>;
}

impl VecConvert<i32> for Vec<usize> {
    fn r_int_convert(self) -> Vec<i32> {
        self.into_iter().map(|x| x as i32).collect()
    }
    fn r_int_convert_shift(self) -> Vec<i32> {
        self.into_iter().map(|x| (x as i32) + 1).collect()
    }
}

impl VecConvert<i32> for &[&usize] {
    fn r_int_convert(self) -> Vec<i32> {
        self.iter().map(|&&x| x as i32).collect()
    }
    fn r_int_convert_shift(self) -> Vec<i32> {
        self.iter().map(|&&x| (x as i32) + 1).collect()
    }
}

impl VecConvert<i32> for Vec<u32> {
    fn r_int_convert(self) -> Vec<i32> {
        self.into_iter().map(|x| x as i32).collect()
    }
    fn r_int_convert_shift(self) -> Vec<i32> {
        self.into_iter().map(|x| (x as i32) + 1).collect()
    }
}

impl VecConvert<usize> for Vec<i32> {
    fn r_int_convert(self) -> Vec<usize> {
        self.into_iter().map(|x| x as usize).collect()
    }
    fn r_int_convert_shift(self) -> Vec<usize> {
        self.into_iter().map(|x| (x - 1) as usize).collect()
    }
}

impl VecConvert<i32> for &[usize] {
    fn r_int_convert(self) -> Vec<i32> {
        self.iter().map(|&x| x as i32).collect()
    }
    fn r_int_convert_shift(self) -> Vec<i32> {
        self.iter().map(|&x| (x as i32) + 1).collect()
    }
}

impl VecConvert<usize> for &[i32] {
    fn r_int_convert(self) -> Vec<usize> {
        self.iter().map(|&x| x as usize).collect()
    }
    fn r_int_convert_shift(self) -> Vec<usize> {
        self.iter().map(|&x| (x - 1) as usize).collect()
    }
}

/// Converts vectors between floating-point types for R compatibility.
pub trait VecFloatConvert<U> {
    /// Converts the R double vectors from/to f64 to f32
    fn r_float_convert(self) -> Vec<U>;
}

impl VecFloatConvert<f32> for Vec<f64> {
    fn r_float_convert(self) -> Vec<f32> {
        self.into_iter().map(|x| x as f32).collect()
    }
}

impl VecFloatConvert<f64> for Vec<f32> {
    fn r_float_convert(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

impl VecFloatConvert<f64> for &[f32] {
    fn r_float_convert(self) -> Vec<f64> {
        self.iter().map(|x| *x as f64).collect()
    }
}

impl VecFloatConvert<f32> for &[f64] {
    fn r_float_convert(self) -> Vec<f32> {
        self.iter().map(|&x| x as f32).collect()
    }
}

////////////////
// R and faer //
////////////////

/// Bridge between faer matrix types and R matrix types.
///
/// Defines how to convert faer matrices to R-compatible arrays.
pub trait FaerRType: SimpleEntity + Copy + Clone + 'static {
    /// Type definition to allow R conversion
    type RType: Copy + Clone;

    /// Transform an faer matrix (f32/f64) into an R matrix (f64)
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self::RType, [usize; 2]>;
}

impl FaerRType for f64 {
    type RType = f64;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)])
    }
}

impl FaerRType for i32 {
    type RType = i32;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)])
    }
}

impl FaerRType for f32 {
    type RType = f64;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<f64, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)] as f64)
    }
}
