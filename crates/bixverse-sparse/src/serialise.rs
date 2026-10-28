//! Byte serialisation for the sparse structures and anything built on them.
//!
//! A blob is a short header followed by a bincode payload. The header is
//! 8 magic bytes naming the structure, a `u32` format version and two bytes
//! holding the element widths of the two data layers, all little-endian.
//! bincode is not self-describing, so the header is the only thing that stops
//! a blob of the wrong structure, an older layout or another float width from
//! decoding into garbage. There is no checksum and no compression.

use bincode::config::{Configuration, standard};
use serde::{Serialize, de::DeserializeOwned};

use crate::errors::SparseErrors;
use crate::sparse::CompressedSparseData2;
use crate::traits::BixverseNumeric;

////////////
// Header //
////////////

/// Version of the byte layout. Bump whenever a change to a serialised struct
/// (reordering, removing or retyping a field) stops older blobs decoding.
/// Nothing checks this automatically.
pub const SERIALISE_FORMAT_VERSION: u32 = 1;

/// Magic bytes of a serialised [`CompressedSparseData2`].
const CSR_MAGIC: &[u8; 8] = b"BXSPCSD2";

/// Header length: magic, version, two element widths.
const HEADER_LEN: usize = 8 + 4 + 2;

/// The one bincode configuration used in both directions. Little-endian,
/// variable-length integers, no decode limit. Varints keep the `usize` shape
/// fields portable and shrink the `u32` indices.
const CONFIG: Configuration = standard();

/// Encode a value behind a header.
///
/// ### Params
///
/// * `magic` - Magic bytes naming the structure
/// * `widths` - Element widths of the value's data layers, in bytes
/// * `value` - The value to encode
///
/// ### Returns
///
/// The header followed by the bincode payload.
pub fn encode_with_header<S>(
    magic: &[u8; 8],
    widths: [u8; 2],
    value: &S,
) -> Result<Vec<u8>, SparseErrors>
where
    S: Serialize,
{
    let payload = bincode::serde::encode_to_vec(value, CONFIG)
        .map_err(|e| SparseErrors::SerialisationFailed(e.to_string()))?;

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&SERIALISE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&widths);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Check the header and decode the payload behind it.
///
/// Checks the magic, the version and the widths, in that order, so the error
/// names the first thing that is actually wrong. Trailing bytes after the
/// payload are an error too: they mean the blob is not what the header says.
///
/// ### Params
///
/// * `magic` - Magic bytes the caller expects
/// * `widths` - Element widths the caller expects
/// * `name` - Structure name, for the error message
/// * `bytes` - The blob
///
/// ### Returns
///
/// The decoded value, or the first mismatch.
pub fn decode_with_header<D>(
    magic: &[u8; 8],
    widths: [u8; 2],
    name: &'static str,
    bytes: &[u8],
) -> Result<D, SparseErrors>
where
    D: DeserializeOwned,
{
    if bytes.len() < 8 || &bytes[..8] != magic {
        return Err(SparseErrors::NotSerialised(name));
    }
    if bytes.len() < HEADER_LEN {
        return Err(SparseErrors::SerialisationFailed(format!(
            "{name}: the header is truncated"
        )));
    }

    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("slice of four bytes"));
    if version != SERIALISE_FORMAT_VERSION {
        return Err(SparseErrors::SerialisedVersionMismatch {
            found: version,
            supported: SERIALISE_FORMAT_VERSION,
        });
    }

    let found = [bytes[12], bytes[13]];
    if found != widths {
        return Err(SparseErrors::SerialisedWidthMismatch {
            expected: widths,
            found,
        });
    }

    let (value, read) = bincode::serde::decode_from_slice(&bytes[HEADER_LEN..], CONFIG)
        .map_err(|e| SparseErrors::SerialisationFailed(format!("{name}: {e}")))?;
    if HEADER_LEN + read != bytes.len() {
        return Err(SparseErrors::SerialisationFailed(format!(
            "{name}: {} trailing bytes after the payload",
            bytes.len() - HEADER_LEN - read
        )));
    }

    Ok(value)
}

/// Element width of a type, as stored in the header.
///
/// ### Returns
///
/// `size_of::<T>()` as a byte.
#[inline]
pub fn elem_width<T>() -> u8 {
    std::mem::size_of::<T>() as u8
}

///////////////////////////
// CompressedSparseData2 //
///////////////////////////

impl<T, U> CompressedSparseData2<T, U>
where
    T: BixverseNumeric + Serialize + DeserializeOwned,
    U: BixverseNumeric + Serialize + DeserializeOwned,
{
    /// Serialise the matrix to bytes.
    ///
    /// ### Returns
    ///
    /// A blob that [`Self::from_bytes`] reads back.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SparseErrors> {
        encode_with_header(CSR_MAGIC, [elem_width::<T>(), elem_width::<U>()], self)
    }

    /// Read a matrix written by [`Self::to_bytes`].
    ///
    /// The decoded structure is validated, since the bytes may come from disk.
    ///
    /// ### Params
    ///
    /// * `bytes` - The blob
    ///
    /// ### Returns
    ///
    /// The matrix, or a header, decode or validation error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SparseErrors> {
        let csr: Self = decode_with_header(
            CSR_MAGIC,
            [elem_width::<T>(), elem_width::<U>()],
            "CompressedSparseData2",
            bytes,
        )?;
        csr.validate()?;
        Ok(csr)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CompressedSparseFormat;

    /// A 3 x 4 CSR with one empty row and a second layer.
    fn small_csr() -> CompressedSparseData2<f32, f64> {
        CompressedSparseData2 {
            data: vec![1.0, 2.0, 3.0, 4.0],
            indices: vec![0, 3, 1, 2],
            indptr: vec![0, 2, 2, 4],
            cs_type: CompressedSparseFormat::Csr,
            data_2: Some(vec![10.0, 20.0, 30.0, 40.0]),
            shape: (3, 4),
        }
    }

    #[test]
    fn test_csr_bytes_round_trip() {
        let csr = small_csr();
        let back = CompressedSparseData2::<f32, f64>::from_bytes(&csr.to_bytes().unwrap()).unwrap();
        assert_eq!(back.data, csr.data);
        assert_eq!(back.indices, csr.indices);
        assert_eq!(back.indptr, csr.indptr);
        assert_eq!(back.data_2, csr.data_2);
        assert_eq!(back.shape, csr.shape);
        assert!(back.cs_type.is_csr());
    }

    #[test]
    fn test_csr_bytes_bad_magic() {
        let mut bytes = small_csr().to_bytes().unwrap();
        bytes[0] = b'X';
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&bytes);
        assert!(matches!(res, Err(SparseErrors::NotSerialised(_))));
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&[1, 2, 3]);
        assert!(matches!(res, Err(SparseErrors::NotSerialised(_))));
    }

    #[test]
    fn test_csr_bytes_version_mismatch() {
        let mut bytes = small_csr().to_bytes().unwrap();
        bytes[8..12].copy_from_slice(&(SERIALISE_FORMAT_VERSION + 1).to_le_bytes());
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(SparseErrors::SerialisedVersionMismatch { found, supported })
                if found == SERIALISE_FORMAT_VERSION + 1 && supported == SERIALISE_FORMAT_VERSION
        ));
    }

    #[test]
    fn test_csr_bytes_width_mismatch() {
        let bytes = small_csr().to_bytes().unwrap();
        let res = CompressedSparseData2::<f64, f64>::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(SparseErrors::SerialisedWidthMismatch { .. })
        ));
    }

    #[test]
    fn test_csr_bytes_truncated_and_trailing() {
        let bytes = small_csr().to_bytes().unwrap();
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&bytes[..bytes.len() - 3]);
        assert!(matches!(res, Err(SparseErrors::SerialisationFailed(_))));

        let mut long = bytes.clone();
        long.push(0);
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&long);
        assert!(matches!(res, Err(SparseErrors::SerialisationFailed(_))));
    }

    #[test]
    fn test_csr_bytes_rejects_invalid_structure() {
        let mut csr = small_csr();
        csr.indices[1] = 9;
        let res = CompressedSparseData2::<f32, f64>::from_bytes(&csr.to_bytes().unwrap());
        assert!(matches!(
            res,
            Err(SparseErrors::SliceIndexOutOfBounds { .. })
        ));
    }
}
