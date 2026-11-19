//! The v5 container: v4 chunks behind a fixed header and a section directory.
//!
//! v5 keeps the v4 chunk codec and changes everything around it:
//!
//! * a hand-laid 128 byte header carrying the feature kind, the norm kind,
//!   the number of raw layers and a hash of the cell IDs;
//! * no bincode main header: the chunk offsets, the library sizes, the
//!   original index lookup and anything added later live in tail sections,
//!   found through a directory the header points at;
//! * offsets are read through the memory map, never decoded into a `Vec`, and
//!   the hash map from original index to chunk is gone. A store written in
//!   index order has no lookup section at all.
//!
//! File layout (all little endian):
//!
//! ```text
//! [0..128)   header, see below
//! [128..)    chunks: u64 length, then the v4 chunk (header plus body)
//! ...        sections, each 8-byte aligned
//! ...        section directory: SECTION_ENTRY_LEN bytes per section
//! ```
//!
//! Header:
//!
//! ```text
//! [0..8)     magic           "SCRNASEQ"
//! [8..12)    version         u32 = 5
//! [12..16)   incompat        u32  bits a reader must understand, see
//!                                 KNOWN_INCOMPAT
//! [16]       cell_based      u8
//! [17]       feature_kind    u8   see FeatureKind
//! [18]       norm_kind       u8   see NormKind
//! [19]       n_layers        u8   raw layers per chunk, 1 to MAX_LAYERS
//! [20..24)   target_size     f32  log-CPM target, TF-IDF scale
//! [24..32)   n_cells         u64
//! [32..40)   n_features      u64
//! [40..48)   n_chunks        u64
//! [48..56)   dir_offset      u64  start of the section directory
//! [56..60)   n_sections      u32
//! [60..64)   ro_compat       u32  bits a reader may ignore but a rewriter
//!                                 must understand, see KNOWN_RO_COMPAT
//! [64..80)   cell_id_hash    [u8; 16], all zero when unset
//! [80..84)   compat          u32  bits anyone may ignore; kept on rewrite
//! [84..88)   tf_idf_scale    f32  TF-IDF scale, 0.0 unless TF-IDF
//! [88..96)   header_hash     u64  xxh3-64 of these 128 bytes (this field
//!                                 zeroed) and the section directory; 0 = none
//! [96..128)  reserved, zero
//! ```
//!
//! Byte 8 holds the version in both this layout and the bincode header of
//! v3 and v4 (a varint below 251 is one byte), so `peek_file_version` reads all
//! three. A v4 reader refuses a v5 file on that byte.
//!
//! Evolving the format without a v6:
//!
//! * a header field or section that changes what a read returns sets an
//!   `incompat` bit or is a section flagged [`SECTION_REQUIRED`]: older builds
//!   refuse the file;
//! * one that older builds may read past but must not carry through a rewrite
//!   (it would go stale) sets a `ro_compat` bit: they read, but refuse every
//!   `set_*` and section rewrite;
//! * anything else is a `compat` bit or an unflagged section. Rewrites patch
//!   only the header fields they know on the raw bytes and carry unknown
//!   sections over, so neither is lost. A section derived from data a rewrite
//!   may change is flagged [`SECTION_DERIVED`] and dropped by builds that do
//!   not know it;
//! * a change inside a chunk needs an `incompat` bit: the chunk header refuses
//!   unknown flags and non-zero reserved bytes, but cannot tell an old reader
//!   more than that.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use indexmap::IndexSet;

use crate::prelude::*;
use crate::store::chunk::{DataLayerReturn, le_u32, le_u64};
use crate::store::reader::{ParallelSparseReader, peek_file_version};
use crate::store::v4::ChunkCodec;

////////////
// Consts //
////////////

/// Length of the fixed v5 file header. Chunks start right after it.
pub const V5_HEADER_LEN: usize = 128;

/// Leading bytes of every bixverse count store.
pub(crate) const V5_MAGIC: &[u8; 8] = b"SCRNASEQ";

/// Bytes per section directory entry: kind u32, flags u32, offset u64,
/// length u64.
const SECTION_ENTRY_LEN: usize = 24;

/// Most raw layers a chunk can carry: layer 0 has its minus-one flag in the
/// chunk flags, layers 1 to 7 in the bitmask byte.
pub const MAX_LAYERS: usize = 8;

/// Header `incompat` bits this build understands. None are defined yet; a
/// file with any other bit set is refused rather than misread.
const KNOWN_INCOMPAT: u32 = 0;

/// Header `ro_compat` bits this build understands. A file with any other bit
/// set opens, but every rewrite refuses it.
const KNOWN_RO_COMPAT: u32 = 0;

/// Section flag: a reader that does not know the section kind must refuse the
/// file. Unflagged unknown sections are skipped on read and kept on rewrite.
pub(crate) const SECTION_REQUIRED: u32 = 1;

/// Section flag: the section is derived from data a rewrite may change, so a
/// build that does not know the kind drops it on rewrite instead of keeping a
/// stale copy.
pub(crate) const SECTION_DERIVED: u32 = 2;

///////////
// Kinds //
///////////

/// What the features of a store are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// Genes (RNA).
    Gene,
    /// Called peaks (ATAC).
    Peak,
    /// Fixed-width genome tiles (ATAC).
    Tile,
    /// Single nucleotide variants (allele counts).
    Snp,
    /// Antibody-derived tags (ADT).
    Protein,
    /// Gene activity: fragments over gene body plus promoter (ATAC). Indexed
    /// by gene, but not RNA counts.
    GeneActivity,
    /// A kind written by a newer bixverse. The kind is a label and does not
    /// change what a read returns, so the file still opens.
    Unknown(u8),
}

impl FeatureKind {
    /// The byte stored in the header.
    ///
    /// ### Returns
    ///
    /// The kind as `u8`.
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            FeatureKind::Gene => 0,
            FeatureKind::Peak => 1,
            FeatureKind::Tile => 2,
            FeatureKind::Snp => 3,
            FeatureKind::Protein => 4,
            FeatureKind::GeneActivity => 5,
            FeatureKind::Unknown(b) => b,
        }
    }

    /// The name [`parse_feature_kind`] reads.
    ///
    /// ### Returns
    ///
    /// The kind as a string.
    pub fn name(self) -> &'static str {
        match self {
            FeatureKind::Gene => "gene",
            FeatureKind::Peak => "peak",
            FeatureKind::Tile => "tile",
            FeatureKind::Snp => "snp",
            FeatureKind::Protein => "protein",
            FeatureKind::GeneActivity => "gene_activity",
            FeatureKind::Unknown(_) => "unknown",
        }
    }

    /// Parse the header byte.
    ///
    /// ### Params
    ///
    /// * `b` - The byte
    ///
    /// ### Returns
    ///
    /// The kind; [`FeatureKind::Unknown`] for a byte this build does not know.
    pub(crate) fn from_u8(b: u8) -> Self {
        match b {
            0 => FeatureKind::Gene,
            1 => FeatureKind::Peak,
            2 => FeatureKind::Tile,
            3 => FeatureKind::Snp,
            4 => FeatureKind::Protein,
            5 => FeatureKind::GeneActivity,
            other => FeatureKind::Unknown(other),
        }
    }
}

/// Parse a feature kind name.
///
/// ### Params
///
/// * `s` - One of `"gene"`, `"peak"`, `"tile"`, `"snp"`, `"protein"` or
///   `"gene_activity"`
///
/// ### Returns
///
/// The [`FeatureKind`], or `None` for anything else.
pub fn parse_feature_kind(s: &str) -> Option<FeatureKind> {
    match s {
        "gene" => Some(FeatureKind::Gene),
        "peak" => Some(FeatureKind::Peak),
        "tile" => Some(FeatureKind::Tile),
        "snp" => Some(FeatureKind::Snp),
        "protein" => Some(FeatureKind::Protein),
        "gene_activity" => Some(FeatureKind::GeneActivity),
        _ => None,
    }
}

/// How the normalised layer of a store is derived from its raw counts.
///
/// Both derived kinds are `ln1p(x / lib * target * w_f)`: log-CPM has
/// `w_f = 1`, TF-IDF has `w_f` the inverse document frequency of feature `f`
/// from the [`SectionKind::Idf`] section and `target` the scale factor. That
/// is TF-IDF method 1 of Signac (`RunTFIDF`) and muon's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// No normalised layer. Reads that ask for it error.
    None,
    /// `ln1p(x / lib * target)`.
    LogCpm,
    /// `ln1p(x / lib * target * idf_f)`.
    TfIdf,
}

impl NormKind {
    /// The byte stored in the header.
    ///
    /// ### Returns
    ///
    /// The kind as `u8`.
    pub(crate) fn to_u8(self) -> u8 {
        match self {
            NormKind::None => 0,
            NormKind::LogCpm => 1,
            NormKind::TfIdf => 2,
        }
    }

    /// Parse the header byte.
    ///
    /// ### Params
    ///
    /// * `b` - The byte
    ///
    /// ### Returns
    ///
    /// The kind, or [`ScDataErrors::V5HeaderInvalid`] for an unknown byte.
    pub(crate) fn from_u8(b: u8) -> Result<Self, ScDataErrors> {
        match b {
            0 => Ok(NormKind::None),
            1 => Ok(NormKind::LogCpm),
            2 => Ok(NormKind::TfIdf),
            other => Err(ScDataErrors::V5HeaderInvalid(format!(
                "unknown norm kind {other}"
            ))),
        }
    }
}

/// What a tail section holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SectionKind {
    /// `n_chunks + 1` u64 offsets relative to the end of the header, the last
    /// one the end of the chunks.
    ChunkOffsets,
    /// Cell files: one u64 library size per chunk, in chunk order.
    LibraryIndex,
    /// Gene files: one u64 library size per original cell index, which gene
    /// chunks recompute their norm from.
    CellLibraryTable,
    /// `(original index u64, chunk u64)` pairs sorted by original index.
    /// Absent when chunk `i` holds original index `i`.
    OriginalIndex,
    /// One f32 inverse document frequency per feature, for
    /// [`NormKind::TfIdf`].
    Idf,
    /// The genome interval index, see `interval_index`.
    IntervalIndex,
    /// One xxh3-64 per chunk over its stored bytes (after the length prefix),
    /// in chunk order. Checked by [`verify_store`].
    ChunkChecksums,
    /// Who wrote the file: `key=value` lines (writer, codec). No timestamp,
    /// so the same input still writes byte-identical stores.
    Provenance,
    /// Free `key=value` lines owned by callers, see [`set_store_metadata`].
    Metadata,
}

impl SectionKind {
    /// The `u32` stored in the directory.
    ///
    /// ### Returns
    ///
    /// The kind code.
    pub(crate) fn code(self) -> u32 {
        match self {
            SectionKind::ChunkOffsets => 1,
            SectionKind::LibraryIndex => 2,
            SectionKind::CellLibraryTable => 3,
            SectionKind::OriginalIndex => 4,
            SectionKind::Idf => 5,
            SectionKind::IntervalIndex => 6,
            SectionKind::ChunkChecksums => 7,
            SectionKind::Provenance => 8,
            SectionKind::Metadata => 9,
        }
    }

    /// Parse a directory code.
    ///
    /// ### Params
    ///
    /// * `code` - The code
    ///
    /// ### Returns
    ///
    /// The kind, `None` for a code this build does not know.
    fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(SectionKind::ChunkOffsets),
            2 => Some(SectionKind::LibraryIndex),
            3 => Some(SectionKind::CellLibraryTable),
            4 => Some(SectionKind::OriginalIndex),
            5 => Some(SectionKind::Idf),
            6 => Some(SectionKind::IntervalIndex),
            7 => Some(SectionKind::ChunkChecksums),
            8 => Some(SectionKind::Provenance),
            9 => Some(SectionKind::Metadata),
            _ => None,
        }
    }

    /// Flags a writer stores with this kind. The kinds that change what a
    /// read returns are required; checksums, provenance and metadata are not.
    /// None is derived: chunks are never rewritten in place.
    ///
    /// ### Returns
    ///
    /// The flags.
    fn flags(self) -> u32 {
        match self {
            SectionKind::ChunkChecksums | SectionKind::Provenance | SectionKind::Metadata => 0,
            _ => SECTION_REQUIRED,
        }
    }
}

////////////
// Header //
////////////

/// The fixed v5 file header, see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct V5FileHeader {
    /// Whether chunks are cells (CSR) rather than features (CSC).
    pub cell_based: bool,
    /// What the features are.
    pub feature_kind: FeatureKind,
    /// How the normalised layer is derived.
    pub norm_kind: NormKind,
    /// Raw layers per chunk.
    pub n_layers: usize,
    /// Log-CPM target size; `0.0` when unknown.
    pub target_size: f32,
    /// TF-IDF scale; `0.0` unless the store is TF-IDF.
    pub tf_idf_scale: f32,
    /// Cells in the store.
    pub n_cells: usize,
    /// Features in the store.
    pub n_features: usize,
    /// Chunks in the file.
    pub n_chunks: usize,
    /// Byte offset of the section directory.
    pub dir_offset: u64,
    /// Entries in the section directory.
    pub n_sections: usize,
    /// xxh3-128 of the cell IDs, all zero when unset.
    pub cell_id_hash: [u8; 16],
    /// Feature bits every reader must understand.
    pub incompat: u32,
    /// Feature bits every rewriter must understand.
    pub ro_compat: u32,
    /// Feature bits anyone may ignore.
    pub compat: u32,
}

/// Byte range of the header hash inside the header.
const HEADER_HASH: std::ops::Range<usize> = 88..96;

impl V5FileHeader {
    /// Write the fields this build knows into header bytes, leaving every
    /// other byte as it is, and seal the header with its hash.
    ///
    /// Rewrites start from the bytes on disk, so reserved bytes a newer build
    /// filled survive.
    ///
    /// ### Params
    ///
    /// * `b` - The header bytes to patch
    /// * `dir` - The section directory the header points at
    pub(crate) fn encode_onto(&self, b: &mut [u8; V5_HEADER_LEN], dir: &[u8]) {
        b[0..8].copy_from_slice(V5_MAGIC);
        b[8..12].copy_from_slice(&SC_FILE_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&self.incompat.to_le_bytes());
        b[16] = self.cell_based as u8;
        b[17] = self.feature_kind.to_u8();
        b[18] = self.norm_kind.to_u8();
        b[19] = self.n_layers as u8;
        b[20..24].copy_from_slice(&self.target_size.to_le_bytes());
        b[24..32].copy_from_slice(&(self.n_cells as u64).to_le_bytes());
        b[32..40].copy_from_slice(&(self.n_features as u64).to_le_bytes());
        b[40..48].copy_from_slice(&(self.n_chunks as u64).to_le_bytes());
        b[48..56].copy_from_slice(&self.dir_offset.to_le_bytes());
        b[56..60].copy_from_slice(&(self.n_sections as u32).to_le_bytes());
        b[60..64].copy_from_slice(&self.ro_compat.to_le_bytes());
        b[64..80].copy_from_slice(&self.cell_id_hash);
        b[80..84].copy_from_slice(&self.compat.to_le_bytes());
        b[84..88].copy_from_slice(&self.tf_idf_scale.to_le_bytes());
        let hash = header_hash(b, dir);
        b[HEADER_HASH].copy_from_slice(&hash.to_le_bytes());
    }

    /// Serialise into a fresh header.
    ///
    /// ### Params
    ///
    /// * `dir` - The section directory the header points at
    ///
    /// ### Returns
    ///
    /// The [`V5_HEADER_LEN`] header bytes.
    pub(crate) fn encode(&self, dir: &[u8]) -> [u8; V5_HEADER_LEN] {
        let mut b = [0u8; V5_HEADER_LEN];
        self.encode_onto(&mut b, dir);
        b
    }

    /// Parse the header at the start of a file.
    ///
    /// ### Params
    ///
    /// * `b` - At least [`V5_HEADER_LEN`] bytes
    ///
    /// ### Returns
    ///
    /// The header, or [`ScDataErrors::V5HeaderInvalid`] for a short buffer,
    /// a foreign magic, another version, an unknown incompat bit, an unknown
    /// norm kind or a layer count outside `1..=MAX_LAYERS`.
    pub(crate) fn parse(b: &[u8]) -> Result<Self, ScDataErrors> {
        let bad = |why: &str| ScDataErrors::V5HeaderInvalid(why.to_string());
        if b.len() < V5_HEADER_LEN {
            return Err(bad("file shorter than the header"));
        }
        if &b[0..8] != V5_MAGIC {
            return Err(bad("magic string mismatch"));
        }
        let version = le_u32(b, 8);
        if version != SC_FILE_VERSION {
            return Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: version,
            });
        }
        let incompat = le_u32(b, 12);
        if incompat & !KNOWN_INCOMPAT != 0 {
            return Err(bad(&format!(
                "incompatible feature bits {incompat:#x}, written by a newer bixverse"
            )));
        }
        let n_layers = b[19] as usize;
        if !(1..=MAX_LAYERS).contains(&n_layers) {
            return Err(bad(&format!("{n_layers} raw layers")));
        }
        let mut cell_id_hash = [0u8; 16];
        cell_id_hash.copy_from_slice(&b[64..80]);
        let f32_at =
            |at: usize| f32::from_le_bytes(b[at..at + 4].try_into().expect("4-byte slice"));
        Ok(Self {
            cell_based: b[16] != 0,
            feature_kind: FeatureKind::from_u8(b[17]),
            norm_kind: NormKind::from_u8(b[18])?,
            n_layers,
            target_size: f32_at(20),
            tf_idf_scale: f32_at(84),
            n_cells: le_u64(b, 24) as usize,
            n_features: le_u64(b, 32) as usize,
            n_chunks: le_u64(b, 40) as usize,
            dir_offset: le_u64(b, 48),
            n_sections: le_u32(b, 56) as usize,
            cell_id_hash,
            incompat,
            ro_compat: le_u32(b, 60),
            compat: le_u32(b, 80),
        })
    }

    /// Error unless this build may rewrite the file.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`ScDataErrors::V5HeaderInvalid`] for an unknown
    /// `ro_compat` bit.
    pub(crate) fn check_rewritable(&self) -> Result<(), ScDataErrors> {
        if self.ro_compat & !KNOWN_RO_COMPAT != 0 {
            return Err(ScDataErrors::V5HeaderInvalid(format!(
                "read-only feature bits {:#x}, written by a newer bixverse: this \
                 build can read the file but not modify it",
                self.ro_compat
            )));
        }
        Ok(())
    }
}

/// xxh3-64 of a header (its hash field zeroed) followed by the directory.
///
/// Never 0, so 0 can mean "no hash".
///
/// ### Params
///
/// * `b` - The header bytes
/// * `dir` - The section directory
///
/// ### Returns
///
/// The hash.
fn header_hash(b: &[u8; V5_HEADER_LEN], dir: &[u8]) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&b[..HEADER_HASH.start]);
    h.update(&[0u8; 8]);
    h.update(&b[HEADER_HASH.end..]);
    h.update(dir);
    h.digest().max(1)
}

/// Check the header hash of a mapped file, if it has one.
///
/// ### Params
///
/// * `file` - The whole file
/// * `h` - Its parsed header
///
/// ### Returns
///
/// `Ok(())`, or [`ScDataErrors::V5HeaderInvalid`] on a mismatch.
pub(crate) fn check_header_hash(file: &[u8], h: &V5FileHeader) -> Result<(), ScDataErrors> {
    let stored = le_u64(file, HEADER_HASH.start);
    if stored == 0 {
        return Ok(());
    }
    let head: &[u8; V5_HEADER_LEN] = file[..V5_HEADER_LEN].try_into().expect("header length");
    if header_hash(head, &file[dir_range(h, file.len())?]) != stored {
        return Err(ScDataErrors::V5HeaderInvalid(
            "header checksum mismatch: the header or section directory is corrupt".into(),
        ));
    }
    Ok(())
}

//////////////
// Sections //
//////////////

/// One entry of the section directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Section {
    /// What the section holds.
    pub kind: SectionKind,
    /// Byte offset in the file.
    pub offset: usize,
    /// Byte length.
    pub len: usize,
}

/// A raw directory entry, known kind or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DirEntry {
    /// Kind code.
    code: u32,
    /// Section flags.
    flags: u32,
    /// Byte offset in the file.
    offset: u64,
    /// Byte length.
    len: u64,
}

/// The byte range of the section directory.
///
/// ### Params
///
/// * `h` - The parsed header
/// * `file_len` - Length of the file
///
/// ### Returns
///
/// `start..end`, or [`ScDataErrors::V5HeaderInvalid`] when it does not fit
/// the file.
fn dir_range(h: &V5FileHeader, file_len: usize) -> Result<std::ops::Range<usize>, ScDataErrors> {
    let dir = h.dir_offset as usize;
    let end = h
        .n_sections
        .checked_mul(SECTION_ENTRY_LEN)
        .and_then(|len| len.checked_add(dir));
    match end {
        Some(end) if dir >= V5_HEADER_LEN && end <= file_len => Ok(dir..end),
        _ => Err(ScDataErrors::V5HeaderInvalid(
            "section directory outside the file".into(),
        )),
    }
}

/// Parse and validate the section directory of a mapped file.
///
/// ### Params
///
/// * `file` - The whole file
/// * `h` - Its parsed header
///
/// ### Returns
///
/// The known sections, or [`ScDataErrors::V5HeaderInvalid`].
pub(crate) fn parse_sections(file: &[u8], h: &V5FileHeader) -> Result<Vec<Section>, ScDataErrors> {
    let range = dir_range(h, file.len())?;
    Ok(parse_directory(&file[range], file.len())?
        .into_iter()
        .filter_map(|e| {
            SectionKind::from_code(e.code).map(|kind| Section {
                kind,
                offset: e.offset as usize,
                len: e.len as usize,
            })
        })
        .collect())
}

/// Parse directory entries.
///
/// Unknown kinds flagged [`SECTION_REQUIRED`] refuse the file; the rest are
/// returned for rewrites to carry. Every section must lie inside the file and
/// no kind may appear twice.
///
/// ### Params
///
/// * `dir` - The directory bytes
/// * `file_len` - Length of the file the sections point into
///
/// ### Returns
///
/// Every entry, or [`ScDataErrors::V5HeaderInvalid`].
fn parse_directory(dir: &[u8], file_len: usize) -> Result<Vec<DirEntry>, ScDataErrors> {
    let bad = |why: String| ScDataErrors::V5HeaderInvalid(why);
    let mut out: Vec<DirEntry> = Vec::with_capacity(dir.len() / SECTION_ENTRY_LEN);
    for at in (0..dir.len()).step_by(SECTION_ENTRY_LEN) {
        let e = DirEntry {
            code: le_u32(dir, at),
            flags: le_u32(dir, at + 4),
            offset: le_u64(dir, at + 8),
            len: le_u64(dir, at + 16),
        };
        if SectionKind::from_code(e.code).is_none() && e.flags & SECTION_REQUIRED != 0 {
            return Err(bad(format!(
                "required section kind {}, written by a newer bixverse",
                e.code
            )));
        }
        let fits = e
            .offset
            .checked_add(e.len)
            .is_some_and(|end| end <= file_len as u64);
        if !fits || e.offset < V5_HEADER_LEN as u64 {
            return Err(bad(format!("section {} outside the file", e.code)));
        }
        if out.iter().any(|o| o.code == e.code) {
            return Err(bad(format!("section {} twice", e.code)));
        }
        out.push(e);
    }
    Ok(out)
}

/// The section of a kind, if present.
///
/// ### Params
///
/// * `sections` - Parsed directory
/// * `kind` - Kind to find
///
/// ### Returns
///
/// The section.
pub(crate) fn find_section(sections: &[Section], kind: SectionKind) -> Option<Section> {
    sections.iter().copied().find(|s| s.kind == kind)
}

/// Check that a section holds exactly `n` fixed-width items.
///
/// ### Params
///
/// * `s` - The section
/// * `n` - Expected items
/// * `width` - Bytes per item
///
/// ### Returns
///
/// `Ok(())`, or [`ScDataErrors::V5HeaderInvalid`].
pub(crate) fn expect_len(s: &Section, n: usize, width: usize) -> Result<(), ScDataErrors> {
    if n.checked_mul(width) != Some(s.len) {
        return Err(ScDataErrors::V5HeaderInvalid(format!(
            "section {:?} is {} bytes, expected {n} items of {width}",
            s.kind, s.len
        )));
    }
    Ok(())
}

/// Encoded tail sections, ready to append after the chunks.
#[derive(Default)]
pub(crate) struct SectionWriter {
    /// `(kind, payload)` in write order.
    pending: Vec<(SectionKind, Vec<u8>)>,
}

impl SectionWriter {
    /// Queue a section of `u64` values.
    ///
    /// ### Params
    ///
    /// * `kind` - Section kind
    /// * `vals` - The values
    pub(crate) fn push_u64(&mut self, kind: SectionKind, vals: impl IntoIterator<Item = u64>) {
        let bytes = vals.into_iter().flat_map(u64::to_le_bytes).collect();
        self.pending.push((kind, bytes));
    }

    /// Queue a section of `f32` values.
    ///
    /// ### Params
    ///
    /// * `kind` - Section kind
    /// * `vals` - The values
    pub(crate) fn push_f32(&mut self, kind: SectionKind, vals: &[f32]) {
        let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.pending.push((kind, bytes));
    }

    /// Queue a section of raw bytes.
    ///
    /// ### Params
    ///
    /// * `kind` - Section kind
    /// * `bytes` - The payload
    pub(crate) fn push_bytes(&mut self, kind: SectionKind, bytes: Vec<u8>) {
        self.pending.push((kind, bytes));
    }

    /// Write the queued sections and the directory at the writer's position.
    ///
    /// Sections start 8-byte aligned so `u64` and `f32` tables can be viewed
    /// in place. A queued section replaces a carried one of the same kind.
    ///
    /// ### Params
    ///
    /// * `w` - Writer positioned at the end of the data
    /// * `carry` - Entries already in the file to list in the new directory
    ///
    /// ### Returns
    ///
    /// `(dir_offset, n_sections, directory bytes)` for the header.
    pub(crate) fn finish<W: Write + Seek>(
        self,
        w: &mut W,
        carry: &[DirEntry],
    ) -> Result<(u64, usize, Vec<u8>), ScDataErrors> {
        let mut entries = carry.to_vec();
        for (kind, bytes) in self.pending {
            let at = align8(w)?;
            w.write_all(&bytes)?;
            entries.retain(|e| e.code != kind.code());
            entries.push(DirEntry {
                code: kind.code(),
                flags: kind.flags(),
                offset: at,
                len: bytes.len() as u64,
            });
        }

        let dir_offset = align8(w)?;
        let mut dir = Vec::with_capacity(entries.len() * SECTION_ENTRY_LEN);
        for e in &entries {
            dir.extend_from_slice(&e.code.to_le_bytes());
            dir.extend_from_slice(&e.flags.to_le_bytes());
            dir.extend_from_slice(&e.offset.to_le_bytes());
            dir.extend_from_slice(&e.len.to_le_bytes());
        }
        w.write_all(&dir)?;
        Ok((dir_offset, entries.len(), dir))
    }
}

/// Pad the writer to an 8-byte boundary.
///
/// ### Params
///
/// * `w` - The writer
///
/// ### Returns
///
/// The aligned position.
fn align8<W: Write + Seek>(w: &mut W) -> Result<u64, ScDataErrors> {
    let pos = w.stream_position()?;
    let pad = (8 - pos % 8) % 8;
    w.write_all(&[0u8; 8][..pad as usize])?;
    Ok(pos + pad)
}

/// Replace or add sections of a finished v5 file, and optionally patch its
/// header.
///
/// Appends the new sections and a fresh directory at the end of the file,
/// then rewrites the header. Unknown sections are carried over unless flagged
/// [`SECTION_DERIVED`]; header bytes this build does not know are kept. A
/// replaced section of a known kind stays as dead bytes, a few per cent of
/// the file at most. The header is written last, so a crash in between leaves
/// the old, consistent file.
///
/// ### Params
///
/// * `path` - The v5 store
/// * `sections` - Sections to write
/// * `patch` - Edits to the header before it is written back
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5 or carries `ro_compat`
/// bits this build does not know.
pub(crate) fn rewrite_sections<P: AsRef<Path>>(
    path: P,
    sections: SectionWriter,
    patch: impl FnOnce(&mut V5FileHeader),
) -> Result<(), ScDataErrors> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    let file_len = f.metadata()?.len() as usize;
    let mut head = [0u8; V5_HEADER_LEN];
    f.read_exact(&mut head)?;
    let mut h = V5FileHeader::parse(&head)?;
    h.check_rewritable()?;

    let range = dir_range(&h, file_len)?;
    let mut dir = vec![0u8; range.len()];
    f.seek(SeekFrom::Start(range.start as u64))?;
    f.read_exact(&mut dir)?;
    let carry: Vec<DirEntry> = parse_directory(&dir, file_len)?
        .into_iter()
        .filter(|e| SectionKind::from_code(e.code).is_some() || e.flags & SECTION_DERIVED == 0)
        .collect();

    f.seek(SeekFrom::End(0))?;
    let (dir_offset, n_sections, dir) = sections.finish(&mut f, &carry)?;
    f.sync_data()?;

    h.dir_offset = dir_offset;
    h.n_sections = n_sections;
    patch(&mut h);
    h.encode_onto(&mut head, &dir);
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&head)?;
    f.sync_data()?;
    Ok(())
}

//////////////////
// Cell ID hash //
//////////////////

/// xxh3-128 over the cell IDs in store order, NUL separated.
///
/// Two stores on the same cell axis (RNA and ATAC of one experiment) carry the
/// same hash; loaders refuse to pair stores whose hashes differ.
///
/// ### Params
///
/// * `ids` - Cell IDs (barcodes), in the order the cells are stored
///
/// ### Returns
///
/// The 16-byte hash, little endian.
pub fn cell_id_hash<S: AsRef<str>>(ids: &[S]) -> [u8; 16] {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for id in ids {
        h.update(id.as_ref().as_bytes());
        h.update(&[0]);
    }
    h.digest128().to_le_bytes()
}

//////////////////////
// Store operations //
//////////////////////

/// Genes (features) summed per batch in [`tf_idf_weights`].
const IDF_GENE_BATCH: usize = 1_024;

/// The inverse document frequency of every feature over a set of cells.
///
/// `idf_f = n / total_f` with `n` the number of cells and `total_f` the summed
/// raw counts of feature `f` over them (Signac `RunTFIDF` method 1, muon
/// `tfidf`). A feature with no counts gets 0; it has no entries to weight.
///
/// ### Params
///
/// * `gene_reader` - Gene-based reader of the store
/// * `cells` - Cells to compute over, `None` for all
///
/// ### Returns
///
/// One weight per feature, or a read error.
pub fn tf_idf_weights(
    gene_reader: &ParallelSparseReader,
    cells: Option<&[usize]>,
) -> Result<Vec<f32>, ScDataErrors> {
    let dims = gene_reader.dims();
    let keep: Option<IndexSet<u32>> = cells.map(|c| c.iter().map(|&i| i as u32).collect());
    let n = keep.as_ref().map_or(dims.total_cells, |k| k.len()) as f64;

    let mut idf = Vec::with_capacity(dims.total_genes);
    let genes: Vec<usize> = (0..dims.total_genes).collect();
    for batch in genes.chunks(IDF_GENE_BATCH) {
        let csc =
            gene_reader.read_genes_sparse::<u32>(batch, keep.as_ref(), &DataLayerReturn::Raw)?;
        for g in 0..batch.len() {
            let (s, e) = (csc.indptr[g] as usize, csc.indptr[g + 1] as usize);
            let total: f64 = csc.data[s..e].iter().map(|&x| x as f64).sum();
            idf.push(if total > 0.0 { (n / total) as f32 } else { 0.0 });
        }
    }
    Ok(idf)
}

/// Read and parse the header of a v5 file this build may rewrite.
///
/// ### Params
///
/// * `path` - The file
///
/// ### Returns
///
/// The header, [`ScDataErrors::FileVersionMismatch`] for another version, or
/// an error for `ro_compat` bits this build does not know.
fn rewritable_header(path: &str) -> Result<V5FileHeader, ScDataErrors> {
    let version = peek_file_version(path)?;
    if version != SC_FILE_VERSION {
        return Err(ScDataErrors::FileVersionMismatch {
            expected: SC_FILE_VERSION,
            found: version,
        });
    }
    let mut head = [0u8; V5_HEADER_LEN];
    std::fs::File::open(path)?.read_exact(&mut head)?;
    let h = V5FileHeader::parse(&head)?;
    h.check_rewritable()?;
    Ok(h)
}

/// Switch the files of a v5 store to TF-IDF normalisation.
///
/// Writes the IDF section and sets the norm kind and the TF-IDF scale in
/// every file's header; no chunk is rewritten, since the norm is derived on
/// read. The log-CPM target stays in the header. Call again after the cell
/// set changes to refresh the IDF.
///
/// ### Params
///
/// * `paths` - The store's files (cell file, gene file)
/// * `idf` - One weight per feature, see [`tf_idf_weights`]
/// * `scale` - Scale factor of the TF-IDF (Signac and muon use `1e4`)
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5, an IDF of the wrong
/// length, or a file that stores a norm verbatim (it would no longer match).
/// Every file is checked before any is changed.
pub fn set_tf_idf(paths: &[&str], idf: &[f32], scale: f32) -> Result<(), ScDataErrors> {
    for &p in paths {
        rewritable_header(p)?;
        let reader = ParallelSparseReader::new(p)?;
        if idf.len() != reader.dims().total_genes {
            return Err(ScDataErrors::InvalidArgument(format!(
                "IDF has {} values for {} features",
                idf.len(),
                reader.dims().total_genes
            )));
        }
        if reader.count_norm_stored_chunks()? > 0 {
            return Err(ScDataErrors::InvalidArgument(format!(
                "{p} stores a norm verbatim, which TF-IDF would not replace"
            )));
        }
    }
    for &p in paths {
        let mut sections = SectionWriter::default();
        sections.push_f32(SectionKind::Idf, idf);
        rewrite_sections(p, sections, |h| {
            h.norm_kind = NormKind::TfIdf;
            h.tf_idf_scale = scale;
        })?;
    }
    Ok(())
}

/// Record the cell IDs of an existing v5 store in its files' headers.
///
/// ### Params
///
/// * `paths` - The store's files (cell file, gene file)
/// * `ids` - Cell IDs in store order
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5 or a cell count other than
/// the store's. Every file is checked before any is changed.
pub fn set_cell_ids<S: AsRef<str>>(paths: &[&str], ids: &[S]) -> Result<(), ScDataErrors> {
    for &p in paths {
        let n = rewritable_header(p)?.n_cells;
        if n != ids.len() {
            return Err(ScDataErrors::InvalidArgument(format!(
                "{} cell IDs for a store of {n} cells",
                ids.len()
            )));
        }
    }
    let hash = cell_id_hash(ids);
    for &p in paths {
        rewrite_sections(p, SectionWriter::default(), |h| h.cell_id_hash = hash)?;
    }
    Ok(())
}

/// Set what the features of an existing v5 store are.
///
/// Header only, no chunk is touched. For ingest paths that write through the
/// generic writers (10x `Peaks`, say) and learn the kind from the input.
///
/// ### Params
///
/// * `paths` - The store's files (cell file, gene file)
/// * `kind` - The feature kind
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5. Every file is checked
/// before any is changed.
pub fn set_feature_kind(paths: &[&str], kind: FeatureKind) -> Result<(), ScDataErrors> {
    for &p in paths {
        rewritable_header(p)?;
    }
    for &p in paths {
        rewrite_sections(p, SectionWriter::default(), |h| h.feature_kind = kind)?;
    }
    Ok(())
}

/// Parse `key=value` lines.
///
/// ### Params
///
/// * `bytes` - The section payload
///
/// ### Returns
///
/// The pairs in file order; lines without `=` are skipped.
pub(crate) fn parse_key_values(bytes: &[u8]) -> Vec<(String, String)> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Encode `key=value` lines.
///
/// ### Params
///
/// * `pairs` - The pairs
///
/// ### Returns
///
/// The payload.
pub(crate) fn encode_key_values(pairs: &[(String, String)]) -> Vec<u8> {
    pairs
        .iter()
        .flat_map(|(k, v)| format!("{k}={v}\n").into_bytes())
        .collect()
}

/// Set caller-owned metadata on every file of a v5 store.
///
/// Keys already present are replaced, others kept. Metadata does not change
/// what a read returns; it is for facts about the store that downstream code
/// needs back on open (an ATAC store's count mode, say).
///
/// ### Params
///
/// * `paths` - The store's files (cell file, gene file)
/// * `entries` - `(key, value)` pairs. Keys may not contain `=` or a line
///   break, values no line break.
///
/// ### Returns
///
/// `Ok(())`, or an error for a bad key or value or a file that is not v5.
/// Every file is checked before any is changed.
pub fn set_store_metadata(paths: &[&str], entries: &[(&str, &str)]) -> Result<(), ScDataErrors> {
    for (k, v) in entries {
        if k.is_empty() || k.contains(['=', '\n', '\r']) || v.contains(['\n', '\r']) {
            return Err(ScDataErrors::InvalidArgument(format!(
                "metadata key {k:?} or its value is not a single line"
            )));
        }
    }
    let mut current = Vec::with_capacity(paths.len());
    for &p in paths {
        rewritable_header(p)?;
        current.push(ParallelSparseReader::new(p)?.store_metadata());
    }
    for (&p, mut pairs) in paths.iter().zip(current) {
        for (k, v) in entries {
            match pairs.iter_mut().find(|(pk, _)| pk == k) {
                Some(slot) => slot.1 = v.to_string(),
                None => pairs.push((k.to_string(), v.to_string())),
            }
        }
        let mut sections = SectionWriter::default();
        sections.push_bytes(SectionKind::Metadata, encode_key_values(&pairs));
        rewrite_sections(p, sections, |_| {})?;
    }
    Ok(())
}

/// The provenance section a writer adds to every v5 file.
///
/// ### Params
///
/// * `codec` - Chunk body codec
///
/// ### Returns
///
/// The payload.
pub(crate) fn provenance(codec: ChunkCodec) -> Vec<u8> {
    encode_key_values(&[
        (
            "writer".into(),
            format!("bixverse-sc-data {}", env!("CARGO_PKG_VERSION")),
        ),
        ("codec".into(), format!("{codec:?}").to_lowercase()),
    ])
}

/// Check every chunk of a v5 file against its stored checksum.
///
/// The header and section directory are checked on every open; this reads
/// every chunk, so it is for an explicit integrity check, not the read path.
///
/// ### Params
///
/// * `path` - The file
///
/// ### Returns
///
/// `true` when every chunk matched, `false` when the file carries no
/// checksums (v4, or a v5 file written without them), or
/// [`ScDataErrors::V4ChunkCorrupt`] naming the first chunk that does not
/// match.
pub fn verify_store(path: &str) -> Result<bool, ScDataErrors> {
    ParallelSparseReader::new(path)?.verify_chunks()
}

/// Error unless every store was written for the same cells.
///
/// ### Params
///
/// * `readers` - Readers over stores meant to share a cell axis
///
/// ### Returns
///
/// `Ok(())` when every store carries the same cell-ID hash, an error when one
/// lacks a hash or they differ.
pub fn check_same_cells(readers: &[&ParallelSparseReader]) -> Result<(), ScDataErrors> {
    let mut hashes = readers.iter().map(|r| r.cell_id_hash());
    let Some(first) = hashes.next() else {
        return Ok(());
    };
    let first = first.ok_or_else(|| {
        ScDataErrors::InvalidArgument("a store without cell IDs cannot be aligned".into())
    })?;
    if hashes.any(|h| h != Some(first)) {
        return Err(ScDataErrors::InvalidArgument(
            "the stores were written for different cells".into(),
        ));
    }
    Ok(())
}

/// Error unless a cell file and a gene file are the two orientations of one
/// store.
///
/// Compares orientation, shape, layers, kinds, norm scale, IDF and cell-ID
/// hash. A pair whose `set_*` was interrupted between its two files, or a
/// gene file from another store of the same shape, fails here instead of
/// reading inconsistent norms.
///
/// ### Params
///
/// * `cells` - The cell-based reader
/// * `genes` - The gene-based reader
///
/// ### Returns
///
/// `Ok(())`, or [`ScDataErrors::StorePairMismatch`] naming the first field
/// that differs.
pub fn check_store_pair(
    cells: &ParallelSparseReader,
    genes: &ParallelSparseReader,
) -> Result<(), ScDataErrors> {
    let differ = |field: &str| Err(ScDataErrors::StorePairMismatch(field.to_string()));
    if !cells.is_cell_based() || genes.is_cell_based() {
        return differ("orientation (expected a cell file and a gene file)");
    }
    let (c, g) = (cells.dims(), genes.dims());
    if (c.total_cells, c.total_genes) != (g.total_cells, g.total_genes) {
        return differ("shape");
    }
    if cells.n_layers() != genes.n_layers() {
        return differ("raw layer count");
    }
    if cells.feature_kind() != genes.feature_kind() {
        return differ("feature kind");
    }
    if cells.norm_kind() != genes.norm_kind() {
        return differ("norm kind");
    }
    if cells.norm_scale().to_bits() != genes.norm_scale().to_bits() {
        return differ("norm target size or TF-IDF scale");
    }
    let bits = |r: &ParallelSparseReader| {
        r.idf()
            .map(|w| w.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
    };
    if bits(cells) != bits(genes) {
        return differ("IDF");
    }
    if cells.cell_id_hash() != genes.cell_id_hash() {
        return differ("cell IDs");
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::chunk::CsrCellChunk;
    use crate::store::convert::{convert_store_to_v5, upgrade_store_to_v5};
    use crate::store::gene_file::write_gene_file;
    use crate::store::merge::{BinMergeTask, gene_store_to_cell_store, merge_sc_bin_files};
    use crate::store::reader::SingleCellReading;
    use crate::store::v4::tests::{
        N_CELLS, N_GENES, TARGET, scratch, synthetic_cells, write_legacy_store,
    };
    use crate::store::writer::CellGeneSparseWriter;

    /// Write the synthetic store as v5, every norm derivable.
    fn v5_store(cell: &str, gene: &str) {
        let mut w = CellGeneSparseWriter::new(cell, true, N_CELLS, N_GENES, TARGET).unwrap();
        for chunk in synthetic_cells() {
            let raw: Vec<u32> = chunk.data_raw.iter().collect();
            let clean =
                CsrCellChunk::from_data(&raw, &chunk.indices, chunk.original_index, TARGET, true);
            w.write_cell_chunk(clean).unwrap();
        }
        w.finalise().unwrap();
        write_gene_file(cell, gene, None, false).unwrap();
    }

    /// Edit the bytes of a file in place.
    fn patch(path: &str, edit: impl FnOnce(&mut Vec<u8>)) {
        let mut b = std::fs::read(path).unwrap();
        edit(&mut b);
        std::fs::write(path, b).unwrap();
    }

    /// Recompute the header hash after a hand edit of the header.
    fn reseal(path: &str) {
        patch(path, |b| {
            let h = V5FileHeader::parse(b).unwrap();
            let dir = b[dir_range(&h, b.len()).unwrap()].to_vec();
            let mut head: [u8; V5_HEADER_LEN] = b[..V5_HEADER_LEN].try_into().unwrap();
            h.encode_onto(&mut head, &dir);
            b[..V5_HEADER_LEN].copy_from_slice(&head);
        });
    }

    /// Append sections of arbitrary kind codes, as a newer build would.
    fn append_raw_sections(path: &str, extra: &[(u32, u32, &[u8])]) {
        patch(path, |b| {
            let mut h = V5FileHeader::parse(b).unwrap();
            let range = dir_range(&h, b.len()).unwrap();
            let mut entries = parse_directory(&b[range], b.len()).unwrap();
            for &(code, flags, bytes) in extra {
                b.resize(b.len().next_multiple_of(8), 0);
                entries.push(DirEntry {
                    code,
                    flags,
                    offset: b.len() as u64,
                    len: bytes.len() as u64,
                });
                b.extend_from_slice(bytes);
            }
            b.resize(b.len().next_multiple_of(8), 0);
            h.dir_offset = b.len() as u64;
            h.n_sections = entries.len();
            let mut dir = Vec::new();
            for e in &entries {
                dir.extend_from_slice(&e.code.to_le_bytes());
                dir.extend_from_slice(&e.flags.to_le_bytes());
                dir.extend_from_slice(&e.offset.to_le_bytes());
                dir.extend_from_slice(&e.len.to_le_bytes());
            }
            b.extend_from_slice(&dir);
            let mut head: [u8; V5_HEADER_LEN] = b[..V5_HEADER_LEN].try_into().unwrap();
            h.encode_onto(&mut head, &dir);
            b[..V5_HEADER_LEN].copy_from_slice(&head);
        });
    }

    /// Directory codes of a file.
    fn section_codes(path: &str) -> Vec<u32> {
        let b = std::fs::read(path).unwrap();
        let h = V5FileHeader::parse(&b).unwrap();
        parse_directory(&b[dir_range(&h, b.len()).unwrap()], b.len())
            .unwrap()
            .iter()
            .map(|e| e.code)
            .collect()
    }

    #[test]
    fn test_rewrite_keeps_unknown_header_bytes_and_sections() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        patch(&s[0], |b| {
            b[80..84].copy_from_slice(&0x4u32.to_le_bytes());
            b[100] = 7;
        });
        reseal(&s[0]);
        append_raw_sections(&s[0], &[(99, 0, b"kept"), (98, SECTION_DERIVED, b"stale")]);

        set_cell_ids(&[&s[0]], &vec!["c"; N_CELLS]).unwrap();
        let b = std::fs::read(&s[0]).unwrap();
        assert_eq!(le_u32(&b, 80), 0x4, "compat bits survive");
        assert_eq!(b[100], 7, "reserved bytes survive");
        let codes = section_codes(&s[0]);
        assert!(codes.contains(&99), "unknown section carried");
        assert!(!codes.contains(&98), "unknown derived section dropped");
        assert!(
            ParallelSparseReader::new(&s[0])
                .unwrap()
                .cell_id_hash()
                .is_some()
        );
    }

    #[test]
    fn test_unknown_required_section_refuses_the_file() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        append_raw_sections(&s[0], &[(97, SECTION_REQUIRED, b"new")]);
        assert!(matches!(
            ParallelSparseReader::new(&s[0]),
            Err(ScDataErrors::V5HeaderInvalid(_))
        ));
    }

    #[test]
    fn test_unknown_ro_compat_reads_but_refuses_rewrites() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        patch(&s[0], |b| b[60..64].copy_from_slice(&1u32.to_le_bytes()));
        reseal(&s[0]);
        let before = std::fs::read(&s[0]).unwrap();
        assert!(ParallelSparseReader::new(&s[0]).is_ok());
        assert!(set_feature_kind(&[&s[0]], FeatureKind::Peak).is_err());
        assert_eq!(std::fs::read(&s[0]).unwrap(), before, "file untouched");
    }

    #[test]
    fn test_unknown_incompat_bit_refuses_the_file() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        patch(&s[0], |b| b[12] = 1);
        assert!(ParallelSparseReader::new(&s[0]).is_err());
    }

    #[test]
    fn test_unknown_feature_kind_still_opens() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        patch(&s[0], |b| b[17] = 77);
        reseal(&s[0]);
        let r = ParallelSparseReader::new(&s[0]).unwrap();
        assert_eq!(r.feature_kind(), FeatureKind::Unknown(77));
        assert_eq!(
            r.read_cell(3).unwrap().indices,
            synthetic_cells()[3].indices
        );
    }

    #[test]
    fn test_header_corruption_is_detected() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        patch(&s[0], |b| b[120] ^= 1);
        assert!(matches!(
            ParallelSparseReader::new(&s[0]),
            Err(ScDataErrors::V5HeaderInvalid(m)) if m.contains("checksum")
        ));
    }

    #[test]
    fn test_verify_store_finds_a_corrupt_chunk() {
        let (_dir, s) = scratch(4);
        v5_store(&s[0], &s[1]);
        assert!(verify_store(&s[0]).unwrap());
        assert!(verify_store(&s[1]).unwrap());
        // the last byte of the first chunk's body
        let r = ParallelSparseReader::new(&s[0]).unwrap();
        let (bytes, at) = r.chunk_bytes(0).unwrap();
        let last = at + 8 + bytes.len() - 1;
        drop(r);
        patch(&s[0], |b| b[last] ^= 0xff);
        assert!(matches!(
            verify_store(&s[0]),
            Err(ScDataErrors::V4ChunkCorrupt(m)) if m.contains("chunk 0")
        ));

        write_legacy_store(SC_FILE_VERSION_V4, &s[2], &s[3], TARGET);
        assert!(!verify_store(&s[2]).unwrap(), "v4 has no checksums");
    }

    #[test]
    fn test_chunk_with_unknown_flags_or_bad_length_errors() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        let r = ParallelSparseReader::new(&s[0]).unwrap();
        let (_, at) = r.chunk_bytes(0).unwrap();
        drop(r);
        let original = std::fs::read(&s[0]).unwrap();

        patch(&s[0], |b| b[at + 8 + 25] |= 0x80);
        assert!(
            ParallelSparseReader::new(&s[0])
                .unwrap()
                .read_cell(0)
                .is_err()
        );

        std::fs::write(&s[0], &original).unwrap();
        patch(&s[0], |b| {
            b[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes())
        });
        assert!(
            ParallelSparseReader::new(&s[0])
                .unwrap()
                .read_cell(0)
                .is_err()
        );
    }

    #[test]
    fn test_store_pair_check() {
        let (_dir, s) = scratch(4);
        v5_store(&s[0], &s[1]);
        let pair = |a: &str, b: &str| {
            check_store_pair(
                &ParallelSparseReader::new(a).unwrap(),
                &ParallelSparseReader::new(b).unwrap(),
            )
        };
        pair(&s[0], &s[1]).unwrap();
        assert!(pair(&s[1], &s[0]).is_err(), "orientation");

        set_cell_ids(&[&s[0]], &vec!["x"; N_CELLS]).unwrap();
        assert!(matches!(
            pair(&s[0], &s[1]),
            Err(ScDataErrors::StorePairMismatch(m)) if m.contains("cell IDs")
        ));
        set_cell_ids(&[&s[1]], &vec!["x"; N_CELLS]).unwrap();
        pair(&s[0], &s[1]).unwrap();

        let genes = ParallelSparseReader::new(&s[1]).unwrap();
        let idf = tf_idf_weights(&genes, None).unwrap();
        drop(genes);
        set_tf_idf(&[&s[0]], &idf, 1e4).unwrap();
        assert!(matches!(
            pair(&s[0], &s[1]),
            Err(ScDataErrors::StorePairMismatch(m)) if m.contains("norm kind")
        ));
    }

    #[test]
    fn test_tf_idf_keeps_the_log_cpm_target() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        let idf = tf_idf_weights(&ParallelSparseReader::new(&s[1]).unwrap(), None).unwrap();
        set_tf_idf(&[&s[0], &s[1]], &idf, 5e3).unwrap();
        let r = ParallelSparseReader::new(&s[0]).unwrap();
        assert_eq!(r.target_size(), None, "not a log-CPM store any more");
        assert_eq!(r.header_scales(), (TARGET, 5e3));
        assert_eq!(r.tf_idf_scale(), Some(5e3));
    }

    #[test]
    fn test_setters_check_every_file_first() {
        let (_dir, s) = scratch(4);
        v5_store(&s[0], &s[1]);
        write_legacy_store(SC_FILE_VERSION_V4, &s[2], &s[3], TARGET);
        let before = std::fs::read(&s[0]).unwrap();
        assert!(matches!(
            set_cell_ids(&[&s[0], &s[3]], &vec!["c"; N_CELLS]),
            Err(ScDataErrors::FileVersionMismatch { found: 4, .. })
        ));
        assert_eq!(std::fs::read(&s[0]).unwrap(), before);
    }

    #[test]
    fn test_metadata_and_provenance() {
        let (_dir, s) = scratch(2);
        v5_store(&s[0], &s[1]);
        set_store_metadata(&[&s[0], &s[1]], &[("count_mode", "insertion"), ("a", "1")]).unwrap();
        set_store_metadata(&[&s[0], &s[1]], &[("count_mode", "fragment")]).unwrap();
        assert!(set_store_metadata(&[&s[0]], &[("bad=key", "x")]).is_err());
        let r = ParallelSparseReader::new(&s[1]).unwrap();
        assert_eq!(
            r.store_metadata(),
            vec![
                ("count_mode".to_string(), "fragment".to_string()),
                ("a".to_string(), "1".to_string())
            ]
        );
        assert!(
            r.provenance()
                .iter()
                .any(|(k, v)| k == "writer" && v.starts_with("bixverse-sc-data"))
        );
    }

    #[test]
    fn test_merge_refuses_mixed_identities_and_keeps_the_kind() {
        let (_dir, s) = scratch(6);
        v5_store(&s[0], &s[1]);
        v5_store(&s[2], &s[3]);
        set_feature_kind(&[&s[2], &s[3]], FeatureKind::Peak).unwrap();
        let task = |p: &str| BinMergeTask {
            exp_id: p.to_string(),
            bin_cells_path: p.to_string(),
            cells_to_keep: (0..10).collect(),
            gene_local_to_universe: (0..N_GENES as u32).map(Some).collect(),
        };
        let merge = |tasks: &[BinMergeTask], renormalise: bool| {
            merge_sc_bin_files(tasks, &s[4], N_GENES, renormalise, TARGET, false)
        };
        assert!(matches!(
            merge(&[task(&s[0]), task(&s[2])], true),
            Err(ScDataErrors::StoreKindMismatch(_))
        ));

        let idf = tf_idf_weights(&ParallelSparseReader::new(&s[3]).unwrap(), None).unwrap();
        set_tf_idf(&[&s[2], &s[3]], &idf, 1e4).unwrap();
        assert!(merge(&[task(&s[2]), task(&s[2])], false).is_err());
        merge(&[task(&s[2]), task(&s[2])], true).unwrap();
        let out = ParallelSparseReader::new(&s[4]).unwrap();
        assert_eq!(out.feature_kind(), FeatureKind::Peak);
        assert_eq!(out.norm_kind(), NormKind::LogCpm);
        assert_eq!(out.dims().total_cells, 20);
    }

    #[test]
    fn test_merge_renormalises_a_store_without_a_norm() {
        let (_dir, s) = scratch(2);
        let mut w = CellGeneSparseWriter::new(&s[0], true, 3, 4, 0.0)
            .unwrap()
            .with_kinds(FeatureKind::Snp, NormKind::None)
            .unwrap();
        for c in 0..3 {
            w.write_cell_layers(c, &[0, 2], &[&[1, 5]], true).unwrap();
        }
        w.finalise().unwrap();
        let task = BinMergeTask {
            exp_id: "a".into(),
            bin_cells_path: s[0].clone(),
            cells_to_keep: vec![0, 2],
            gene_local_to_universe: vec![Some(0), Some(1), Some(2), Some(3)],
        };
        merge_sc_bin_files(&[task], &s[1], 4, true, TARGET, false).unwrap();
        let out = ParallelSparseReader::new(&s[1]).unwrap();
        assert_eq!(out.feature_kind(), FeatureKind::Snp);
        assert_eq!(
            out.read_cell(1)
                .unwrap()
                .data_raw
                .iter()
                .collect::<Vec<_>>(),
            [1, 5]
        );
    }

    #[test]
    fn test_transpose_keeps_a_tf_idf_store() {
        let (_dir, s) = scratch(3);
        v5_store(&s[0], &s[1]);
        set_feature_kind(&[&s[0], &s[1]], FeatureKind::Tile).unwrap();
        let idf = tf_idf_weights(&ParallelSparseReader::new(&s[1]).unwrap(), None).unwrap();
        set_tf_idf(&[&s[0], &s[1]], &idf, 1e4).unwrap();
        gene_store_to_cell_store(&s[1], &s[2], 64, 8, 0).unwrap();

        let (a, b) = (
            ParallelSparseReader::new(&s[0]).unwrap(),
            ParallelSparseReader::new(&s[2]).unwrap(),
        );
        assert_eq!(b.feature_kind(), FeatureKind::Tile);
        assert_eq!(b.norm_kind(), NormKind::TfIdf);
        assert_eq!(b.header_scales(), a.header_scales());
        assert_eq!(b.idf(), a.idf());
        let cells: Vec<usize> = (0..N_CELLS).collect();
        for (x, y) in a
            .read_cells_parallel(&cells)
            .unwrap()
            .iter()
            .zip(b.read_cells_parallel(&cells).unwrap())
        {
            assert_eq!(x.indices, y.indices);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x.data_norm), bits(&y.data_norm));
        }
    }

    #[test]
    fn test_convert_needs_both_gene_paths() {
        let (_dir, s) = scratch(4);
        write_legacy_store(SC_FILE_VERSION_V4, &s[0], &s[1], TARGET);
        assert!(matches!(
            convert_store_to_v5(&s[0], Some(&s[1]), &s[2], None, ChunkCodec::Lz4, false),
            Err(ScDataErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_upgrade_keeps_a_backup_and_cleans_stale_tmp_files() {
        let (_dir, s) = scratch(2);
        write_legacy_store(SC_FILE_VERSION_V4, &s[0], &s[1], TARGET);
        std::fs::write(format!("{}.v5.tmp", s[1]), b"left over").unwrap();
        upgrade_store_to_v5(&s[0], Some(&s[1]), ChunkCodec::Lz4, true, false).unwrap();
        for p in &s {
            assert_eq!(peek_file_version(p).unwrap(), SC_FILE_VERSION);
            assert_eq!(peek_file_version(format!("{p}.v4.bak")).unwrap(), 4);
            assert!(!std::path::Path::new(&format!("{p}.v5.tmp")).exists());
        }
        check_store_pair(
            &ParallelSparseReader::new(&s[0]).unwrap(),
            &ParallelSparseReader::new(&s[1]).unwrap(),
        )
        .unwrap();
        // a second run on a v5 store does nothing
        let before = std::fs::read(&s[0]).unwrap();
        upgrade_store_to_v5(&s[0], Some(&s[1]), ChunkCodec::Lz4, true, false).unwrap();
        assert_eq!(std::fs::read(&s[0]).unwrap(), before);
    }

    #[test]
    fn test_upgrade_resumes_after_only_the_cell_file_was_swapped() {
        let (_dir, s) = scratch(3);
        write_legacy_store(SC_FILE_VERSION_V4, &s[0], &s[1], TARGET);
        let old_genes = ParallelSparseReader::open_any(&s[1]).unwrap();
        let want: Vec<Vec<u32>> = (0..N_GENES)
            .map(|g| old_genes.read_gene(g).unwrap().indices)
            .collect();
        drop(old_genes);
        // what an interrupted upgrade leaves: a v5 cell file next to a v4 gene file
        convert_store_to_v5(&s[0], None, &s[2], None, ChunkCodec::Lz4, false).unwrap();
        std::fs::rename(&s[2], &s[0]).unwrap();

        upgrade_store_to_v5(&s[0], Some(&s[1]), ChunkCodec::Lz4, false, false).unwrap();
        let genes = ParallelSparseReader::new(&s[1]).unwrap();
        assert_eq!(genes.file_version(), SC_FILE_VERSION);
        for (g, w) in want.iter().enumerate() {
            assert_eq!(&genes.read_gene(g).unwrap().indices, w);
        }
        check_store_pair(&ParallelSparseReader::new(&s[0]).unwrap(), &genes).unwrap();
    }

    #[test]
    fn test_gene_file_refuses_unsorted_cells() {
        let (_dir, s) = scratch(2);
        let mut w = CellGeneSparseWriter::new(&s[0], true, 2, 4, TARGET).unwrap();
        w.write_cell_layers(0, &[0, 3], &[&[1, 2]], true).unwrap();
        w.write_cell_layers(1, &[2, 1], &[&[1, 2]], true).unwrap();
        w.finalise().unwrap();
        assert!(matches!(
            write_gene_file(&s[0], &s[1], None, false),
            Err(ScDataErrors::InvalidArgument(m)) if m.contains("cell 1")
        ));
    }

    #[test]
    fn test_feature_kind_byte_and_name_round_trip() {
        for b in 0..=5u8 {
            let k = FeatureKind::from_u8(b);
            assert_eq!(k.to_u8(), b);
            assert_eq!(parse_feature_kind(k.name()), Some(k));
        }
        assert_eq!(FeatureKind::from_u8(200), FeatureKind::Unknown(200));
        assert_eq!(FeatureKind::Unknown(200).to_u8(), 200);
    }
}
