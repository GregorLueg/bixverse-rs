//! Save and load compressed sparse data formats to and from disk

use bincode::{
    Decode, Encode, config, decode_from_slice, decode_from_std_read, encode_into_std_write,
    encode_to_vec,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::prelude::*;

////////////
// Header //
////////////

/// The magic bits for the meta cell part
const MC_SPARSE_MAGIC: [u8; 8] = *b"BIXVMCSP";

/// Fixed 64-byte header for a whole-blob metacell sparse dump.
#[derive(Encode, Decode)]
struct McSparseHeader {
    /// Magic string as bytes to recognise the file
    magic: [u8; 8],
    /// Version of the file
    version: u32,
    /// Which compressed sparse format it is -> 0 = Csc, 1 = Csr
    cs_type: u8,
    /// Is the second data layer populated
    has_data2: u8,
    /// Number of rows
    nrow: u64,
    /// Number of columns
    ncol: u64,
    /// Number of non-zero entries
    nnz: u64,
    /// 26 additional reserved bytes for the future
    _reserved: [u8; 26],
}

impl<T, U> CompressedSparseData2<T, U>
where
    T: BixverseNumeric,
    U: BixverseNumeric,
{
    /// Write the entire structure to disk. Not for partial/streamed access.
    ///
    /// ### Params
    ///
    /// * `path` - Where to save the file to
    pub fn write_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<(), BixverseErrors>
    where
        T: Encode,
        U: Encode,
    {
        let header_cfg = config::standard().with_fixed_int_encoding();
        let body_cfg = config::standard();

        let header = McSparseHeader {
            magic: MC_SPARSE_MAGIC,
            version: MC_SPARSE_VERSION,
            cs_type: match self.cs_type {
                CompressedSparseFormat::Csc => 0,
                CompressedSparseFormat::Csr => 1,
            },
            has_data2: self.data_2.is_some() as u8,
            nrow: self.shape.0 as u64,
            ncol: self.shape.1 as u64,
            nnz: self.indices.len() as u64,
            _reserved: [0; 26],
        };

        let mut writer = BufWriter::new(File::create(path)?);

        let mut header_bytes =
            encode_to_vec(&header, header_cfg).map_err(|_| BixverseErrors::HeaderEncodeFailed)?;
        header_bytes.resize(64, 0); // fixed-int encoding is already 64; defensive
        writer.write_all(&header_bytes)?;

        encode_into_std_write(&self.indptr, &mut writer, body_cfg)
            .map_err(|_| BixverseErrors::SerialisationFailed)?;
        encode_into_std_write(&self.indices, &mut writer, body_cfg)
            .map_err(|_| BixverseErrors::SerialisationFailed)?;
        encode_into_std_write(&self.data, &mut writer, body_cfg)
            .map_err(|_| BixverseErrors::SerialisationFailed)?;
        encode_into_std_write(&self.data_2, &mut writer, body_cfg)
            .map_err(|_| BixverseErrors::SerialisationFailed)?;

        writer.flush()?;
        Ok(())
    }

    /// Read a structure previously written by `write_to_disk`.
    ///
    /// ### Params
    ///
    /// * `path` - Path were the file is stored
    pub fn read_from_disk<P: AsRef<Path>>(path: P) -> Result<Self, BixverseErrors>
    where
        T: Decode<()>,
        U: Decode<()>,
    {
        let header_cfg = config::standard().with_fixed_int_encoding();
        let body_cfg = config::standard();

        let mut reader = BufReader::new(File::open(path)?);

        let mut header_buf = [0u8; 64];
        reader.read_exact(&mut header_buf)?;
        let (header, _) = decode_from_slice::<McSparseHeader, _>(&header_buf, header_cfg)
            .map_err(|_| BixverseErrors::HeaderDecodeFailed)?;

        if header.magic != MC_SPARSE_MAGIC {
            return Err(BixverseErrors::HeaderDecodeFailed);
        }
        if header.version != MC_SPARSE_VERSION {
            return Err(BixverseErrors::FileVersionMismatch {
                expected: MC_SPARSE_VERSION,
                found: header.version,
            });
        }

        let cs_type = match header.cs_type {
            0 => CompressedSparseFormat::Csc,
            1 => CompressedSparseFormat::Csr,
            _ => return Err(BixverseErrors::HeaderDecodeFailed),
        };

        let indptr: Vec<u32> = decode_from_std_read(&mut reader, body_cfg)
            .map_err(|_| BixverseErrors::DeserialisationFailed)?;
        let indices: Vec<u32> = decode_from_std_read(&mut reader, body_cfg)
            .map_err(|_| BixverseErrors::DeserialisationFailed)?;
        let data: Vec<T> = decode_from_std_read(&mut reader, body_cfg)
            .map_err(|_| BixverseErrors::DeserialisationFailed)?;
        let data_2: Option<Vec<U>> = decode_from_std_read(&mut reader, body_cfg)
            .map_err(|_| BixverseErrors::DeserialisationFailed)?;

        let out = Self {
            data,
            indices,
            indptr,
            cs_type,
            data_2,
            shape: (header.nrow as usize, header.ncol as usize),
        };
        out.assert_invariants();
        Ok(out)
    }
}
