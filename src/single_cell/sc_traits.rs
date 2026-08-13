//! Contains traits and their implementations, specifically designed for the
//! single cell workflows in this crate.

use bincode::{Decode, Encode};
use half::f16;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::iter::Sum;

/////////
// F16 //
/////////

/// A serialisable wrapper for IEEE 754 half-precision floats.
///
/// Stores the raw bits as u16 for compatibility with bincode and serde.
#[derive(Encode, Decode, Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub struct F16(u16);

/////////////////
// Conversions //
/////////////////

impl From<f16> for F16 {
    fn from(f: f16) -> Self {
        F16(f.to_bits())
    }
}

impl From<F16> for f16 {
    fn from(f: F16) -> Self {
        f16::from_bits(f.0)
    }
}

//////////////////
// Iterator sum //
//////////////////

impl Sum for F16 {
    fn sum<I: Iterator<Item = F16>>(iter: I) -> Self {
        let sum: f16 = iter.map(f16::from).sum();
        F16::from(sum)
    }
}

impl<'a> Sum<&'a F16> for F16 {
    fn sum<I: Iterator<Item = &'a F16>>(iter: I) -> Self {
        let sum: f16 = iter.map(|&f| f16::from(f)).sum();
        F16::from(sum)
    }
}

//////////////
// Math ops //
//////////////

impl std::ops::Add for F16 {
    type Output = F16;

    fn add(self, other: F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(other);
        F16::from(a + b)
    }
}

impl<'a> std::ops::Add<&'a F16> for F16 {
    type Output = F16;

    fn add(self, other: &'a F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(*other);
        F16::from(a + b)
    }
}

impl std::ops::Mul for F16 {
    type Output = F16;

    fn mul(self, other: F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(other);
        F16::from(a * b)
    }
}

impl<'a> std::ops::Mul<&'a F16> for F16 {
    type Output = F16;

    fn mul(self, other: &'a F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(*other);
        F16::from(a * b)
    }
}

impl std::ops::Sub for F16 {
    type Output = F16;
    fn sub(self, other: F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(other);
        F16::from(a - b)
    }
}

impl<'a> std::ops::Sub<&'a F16> for F16 {
    type Output = F16;
    fn sub(self, other: &'a F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(*other);
        F16::from(a - b)
    }
}

impl std::ops::AddAssign for F16 {
    fn add_assign(&mut self, other: F16) {
        let a = f16::from(*self);
        let b = f16::from(other);
        *self = F16::from(a + b);
    }
}

impl<'a> std::ops::AddAssign<&'a F16> for F16 {
    fn add_assign(&mut self, other: &'a F16) {
        let a = f16::from(*self);
        let b = f16::from(*other);
        *self = F16::from(a + b);
    }
}

impl std::ops::DivAssign for F16 {
    fn div_assign(&mut self, other: F16) {
        let a = f16::from(*self);
        let b = f16::from(other);
        *self = F16::from(a / b);
    }
}

impl<'a> std::ops::DivAssign<&'a F16> for F16 {
    fn div_assign(&mut self, other: &'a F16) {
        let a = f16::from(*self);
        let b = f16::from(*other);
        *self = F16::from(a / b);
    }
}

impl std::ops::Div for F16 {
    type Output = F16;
    fn div(self, other: F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(other);
        F16::from(a / b)
    }
}

impl<'a> std::ops::Div<&'a F16> for F16 {
    type Output = F16;
    fn div(self, other: &'a F16) -> F16 {
        let a = f16::from(self);
        let b = f16::from(*other);
        F16::from(a / b)
    }
}

//////////////
// Equality //
//////////////

/// Equality is reflexive, including for NaN, which IEEE floats are not.
///
/// [F16] is a storage newtype for the binary format rather than a float proper,
/// and it claims both [Eq] and [Ord]. [Eq] requires `a == a` for every value,
/// and [Ord::cmp] already reports `Equal` for two NaNs, so returning `false`
/// here left the two disagreeing: a NaN compared unequal to itself while
/// sorting as equal to itself. Values arrive from disk through
/// [F16::from_le_bytes] unvalidated, so any `sort`, `dedup`, `BTreeMap` or
/// `binary_search` over them could see that inconsistency.
impl PartialEq for F16 {
    fn eq(&self, other: &Self) -> bool {
        let a = f16::from(*self);
        let b = f16::from(*other);

        if a.is_nan() && b.is_nan() {
            true
        } else {
            a == b
        }
    }
}

impl Eq for F16 {}

impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F16 {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = half::f16::from(*self);
        let b = half::f16::from(*other);

        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.partial_cmp(&b).unwrap(),
        }
    }
}

impl F16 {
    /// Get the raw bits representation
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Create from raw bits
    pub fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    /// From f32
    pub fn from_f32(f: f32) -> Self {
        F16::from(f16::from_f32(f))
    }

    /// Convert to little-endian bytes
    pub fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Create from little-endian bytes
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        F16(u16::from_le_bytes(bytes))
    }

    /// From F16 to f32
    pub fn to_f32(self) -> f32 {
        f16::from_bits(self.0).to_f32()
    }

    /// From F16 fo f64
    pub fn to_f64(self) -> f64 {
        f16::from_bits(self.0).to_f32() as f64
    }
}

///////////////////////
// f32 and u16 stuff //
///////////////////////

/// Trait for types that can be converted to f32 and u16.
pub trait FloatAndUInt: Copy {
    /// Transform u16 to f32 for fast conversions
    fn to_f32(self) -> f32;

    /// Transform u32/f32 to u16, **saturating** at `u16::MAX`.
    ///
    /// Prefer [`Self::to_u32`] plus `RawCounts::from_u32_auto` on any path that
    /// reaches the binary format: that narrows to u16 only when every value
    /// fits, instead of silently clipping high-expression genes.
    fn to_u16(self) -> u16;

    /// Transform f32/u16 to u32
    fn to_u32(self) -> u32;
}

impl FloatAndUInt for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_u16(self) -> u16 {
        self.min(u16::MAX as i32).max(0) as u16
    }
    fn to_u32(self) -> u32 {
        self.max(0) as u32
    }
}

impl FloatAndUInt for u32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_u16(self) -> u16 {
        self.min(u16::MAX as u32) as u16
    }
    fn to_u32(self) -> u32 {
        self
    }
}

impl FloatAndUInt for usize {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_u16(self) -> u16 {
        self.min(u16::MAX as usize) as u16
    }
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl FloatAndUInt for u16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_u16(self) -> u16 {
        self
    }
    fn to_u32(self) -> u32 {
        self as u32
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Regression: `PartialEq` returned false for two NaNs while `Ord::cmp`
    /// returned `Equal`, so `Eq` was claimed on a non-reflexive `PartialEq`.
    /// Any sort or map keyed on `F16` could see the two disagree.
    #[test]
    fn f16_equality_is_reflexive_and_agrees_with_ord() {
        let nan = F16::from_f32(f32::NAN);
        let one = F16::from_f32(1.0);

        assert_eq!(nan, nan, "Eq requires a == a for every value");
        assert_eq!(nan.cmp(&nan), Ordering::Equal);
        assert_eq!(one, one);

        // The two traits must never disagree on any pair.
        for (a, b) in [(nan, nan), (nan, one), (one, nan), (one, one)] {
            assert_eq!(
                a == b,
                a.cmp(&b) == Ordering::Equal,
                "PartialEq and Ord disagree"
            );
        }
    }

    /// NaN sorts after every real value, and sorting must not panic.
    #[test]
    fn f16_sorts_with_nan_last() {
        let mut vals = [
            F16::from_f32(2.0),
            F16::from_f32(f32::NAN),
            F16::from_f32(1.0),
        ];
        vals.sort();

        assert_relative_eq!(vals[0].to_f32(), 1.0, epsilon = 1e-3);
        assert_relative_eq!(vals[1].to_f32(), 2.0, epsilon = 1e-3);
        assert!(vals[2].to_f32().is_nan());
    }

    /// The binary format depends on this pair round-tripping exactly.
    #[test]
    fn f16_le_bytes_round_trip_preserves_bits() {
        for v in [0.0_f32, 1.0, -1.5, 65504.0, f32::INFINITY] {
            let original = F16::from_f32(v);
            let restored = F16::from_le_bytes(original.to_le_bytes());
            assert_eq!(original.to_bits(), restored.to_bits());
        }

        let nan = F16::from_f32(f32::NAN);
        assert!(F16::from_le_bytes(nan.to_le_bytes()).to_f32().is_nan());
    }

    /// `from_bits` and `to_bits` are inverse for every representable pattern.
    #[test]
    fn f16_bits_round_trip() {
        for bits in [0u16, 0x3C00, 0xBC00, 0x7BFF] {
            assert_eq!(F16::from_bits(bits).to_bits(), bits);
        }
    }

    /// Half precision carries about three decimal digits, so 0.1 does not
    /// survive exactly. Pinning that keeps anyone from assuming f32 fidelity.
    #[test]
    fn f16_conversion_loses_precision_predictably() {
        let v = F16::from_f32(0.1);
        assert_relative_eq!(v.to_f32(), 0.1, epsilon = 1e-3);
        assert_relative_eq!(v.to_f64(), 0.1, epsilon = 1e-3);
        assert_ne!(v.to_f32(), 0.1_f32, "0.1 is not representable in f16");
    }

    /// `Sum` accumulates in f16 rather than widening, and the error compounds
    /// badly with length. Once the running total passes 64 the f16 step is
    /// 0.0625, so adding 0.1 rounds up to two steps every time and the sum
    /// drifts *upwards*: 1000 tenths land on 105.19, not 100, a 5.2% overshoot.
    ///
    /// This is pinned rather than tolerated because
    /// [in_memory_io](crate::single_cell::sc_data::in_memory_io) computes
    /// `avg_exp` this way over a whole gene column, so a gene expressed in
    /// thousands of cells carries that bias.
    #[test]
    fn f16_sum_drifts_upwards_on_long_columns() {
        // Exact while the running total stays small.
        let exact: F16 = (0..4).map(|_| F16::from_f32(0.25)).sum();
        assert_relative_eq!(exact.to_f32(), 1.0, epsilon = 1e-3);

        let drifting: F16 = (0..1000).map(|_| F16::from_f32(0.1)).sum();
        assert_relative_eq!(drifting.to_f32(), 105.1875, epsilon = 1e-3);
        assert!(
            drifting.to_f32() > 100.0,
            "the drift is upwards, not symmetric rounding"
        );
    }

    /// Arithmetic round-trips through f16 at every step.
    #[test]
    fn f16_arithmetic_round_trips_through_half_precision() {
        let a = F16::from_f32(1.5);
        let b = F16::from_f32(2.5);

        assert_relative_eq!((a + b).to_f32(), 4.0, epsilon = 1e-3);
        assert_relative_eq!((b - a).to_f32(), 1.0, epsilon = 1e-3);
        assert_relative_eq!((a * b).to_f32(), 3.75, epsilon = 1e-3);
        assert_relative_eq!((b / a).to_f32(), 5.0 / 3.0, epsilon = 1e-3);

        let mut acc = a;
        acc += b;
        assert_relative_eq!(acc.to_f32(), 4.0, epsilon = 1e-3);
    }
}
