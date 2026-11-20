//! Reading what an ATAC experiment ships: fragment files (BGZF, tabix), GTF
//! gene annotation and the 10x multiome peak matrix.

pub mod annotation;
pub mod fragments;
pub mod merge;
#[cfg(feature = "h5")]
pub mod tenx_peaks;

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::read::MultiGzDecoder;

/// Open a text file that may be gzip or BGZF compressed, decided by its magic
/// bytes rather than its extension.
///
/// ### Params
///
/// * `path` - The file
///
/// ### Returns
///
/// A buffered reader over the text, or an I/O error.
pub(crate) fn open_text(path: &Path) -> std::io::Result<Box<dyn BufRead>> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 2];
    let gz = file.read(&mut magic)? == 2 && magic == [0x1f, 0x8b];
    file.seek(SeekFrom::Start(0))?;
    Ok(if gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    })
}
