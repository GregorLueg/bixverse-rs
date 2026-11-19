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
//! [60..64)   reserved
//! [64..80)   cell_id_hash    [u8; 16], all zero when unset
//! [80..128)  reserved
//! ```
//!
//! Byte 8 holds the version in both this layout and the bincode header of
//! v3 and v4 (a varint below 251 is one byte), so `peek_file_version` reads all
//! three. A v4 reader refuses a v5 file on that byte.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use indexmap::IndexSet;

use crate::prelude::*;
use crate::store::chunk::{DataLayerReturn, le_u32, le_u64};
use crate::store::reader::ParallelSparseReader;

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
/// file with any bit set is refused rather than misread.
const KNOWN_INCOMPAT: u32 = 0;

/// Section flag: a reader that does not know the section kind must refuse the
/// file. Unflagged unknown sections are skipped.
pub(crate) const SECTION_REQUIRED: u32 = 1;

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
            0 => Ok(FeatureKind::Gene),
            1 => Ok(FeatureKind::Peak),
            2 => Ok(FeatureKind::Tile),
            3 => Ok(FeatureKind::Snp),
            4 => Ok(FeatureKind::Protein),
            5 => Ok(FeatureKind::GeneActivity),
            other => Err(ScDataErrors::V5HeaderInvalid(format!(
                "unknown feature kind {other}"
            ))),
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
            _ => None,
        }
    }

    /// Flags a writer stores with this kind. Every kind defined so far changes
    /// what a read returns, so all are required.
    ///
    /// ### Returns
    ///
    /// The flags.
    fn flags(self) -> u32 {
        SECTION_REQUIRED
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
    /// Log-CPM target size or TF-IDF scale; `0.0` when unknown.
    pub target_size: f32,
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
}

impl V5FileHeader {
    /// Serialise into the fixed layout.
    ///
    /// ### Returns
    ///
    /// The [`V5_HEADER_LEN`] header bytes.
    pub(crate) fn encode(&self) -> [u8; V5_HEADER_LEN] {
        let mut b = [0u8; V5_HEADER_LEN];
        b[0..8].copy_from_slice(V5_MAGIC);
        b[8..12].copy_from_slice(&SC_FILE_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&0u32.to_le_bytes());
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
        b[64..80].copy_from_slice(&self.cell_id_hash);
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
    /// kind or a layer count outside `1..=MAX_LAYERS`.
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
        Ok(Self {
            cell_based: b[16] != 0,
            feature_kind: FeatureKind::from_u8(b[17])?,
            norm_kind: NormKind::from_u8(b[18])?,
            n_layers,
            target_size: f32::from_le_bytes(b[20..24].try_into().expect("4-byte slice")),
            n_cells: le_u64(b, 24) as usize,
            n_features: le_u64(b, 32) as usize,
            n_chunks: le_u64(b, 40) as usize,
            dir_offset: le_u64(b, 48),
            n_sections: le_u32(b, 56) as usize,
            cell_id_hash,
        })
    }
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
    parse_directory(&file[range], file.len())
}

/// Parse directory entries.
///
/// Unknown kinds without [`SECTION_REQUIRED`] are dropped, unknown required
/// kinds refuse the file. Every section must lie inside the file.
///
/// ### Params
///
/// * `dir` - The directory bytes
/// * `file_len` - Length of the file the sections point into
///
/// ### Returns
///
/// The known sections, or [`ScDataErrors::V5HeaderInvalid`].
fn parse_directory(dir: &[u8], file_len: usize) -> Result<Vec<Section>, ScDataErrors> {
    let bad = |why: String| ScDataErrors::V5HeaderInvalid(why);
    let mut out = Vec::with_capacity(dir.len() / SECTION_ENTRY_LEN);
    for at in (0..dir.len()).step_by(SECTION_ENTRY_LEN) {
        let (code, flags) = (le_u32(dir, at), le_u32(dir, at + 4));
        let (offset, len) = (le_u64(dir, at + 8) as usize, le_u64(dir, at + 16) as usize);
        let Some(kind) = SectionKind::from_code(code) else {
            if flags & SECTION_REQUIRED != 0 {
                return Err(bad(format!(
                    "required section kind {code}, written by a newer bixverse"
                )));
            }
            continue;
        };
        if offset.checked_add(len).is_none_or(|e| e > file_len) || offset < V5_HEADER_LEN {
            return Err(bad(format!("section {code} outside the file")));
        }
        if out.iter().any(|s: &Section| s.kind == kind) {
            return Err(bad(format!("section {code} twice")));
        }
        out.push(Section { kind, offset, len });
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
    /// in place.
    ///
    /// ### Params
    ///
    /// * `w` - Writer positioned at the end of the data
    /// * `carry` - Sections already in the file to list in the new directory
    ///
    /// ### Returns
    ///
    /// `(dir_offset, n_sections)` for the header.
    pub(crate) fn finish<W: Write + Seek>(
        self,
        w: &mut W,
        carry: &[(Section, u32)],
    ) -> Result<(u64, usize), ScDataErrors> {
        let mut entries: Vec<(u32, u32, u64, u64)> = carry
            .iter()
            .map(|&(s, flags)| (s.kind.code(), flags, s.offset as u64, s.len as u64))
            .collect();

        for (kind, bytes) in self.pending {
            let at = align8(w)?;
            w.write_all(&bytes)?;
            entries.retain(|e| e.0 != kind.code());
            entries.push((kind.code(), kind.flags(), at, bytes.len() as u64));
        }

        let dir = align8(w)?;
        for (code, flags, offset, len) in &entries {
            w.write_all(&code.to_le_bytes())?;
            w.write_all(&flags.to_le_bytes())?;
            w.write_all(&offset.to_le_bytes())?;
            w.write_all(&len.to_le_bytes())?;
        }
        Ok((dir, entries.len()))
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
/// then rewrites the header. The superseded sections and directory stay as
/// dead bytes, a few per cent of the file at most. The header is written last,
/// so a crash in between leaves the old, consistent file.
///
/// ### Params
///
/// * `path` - The v5 store
/// * `sections` - Sections to write
/// * `patch` - Edits to the header before it is written back
///
/// ### Returns
///
/// `Ok(())`, or an error for a file that is not v5.
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

    let range = dir_range(&h, file_len)?;
    let mut dir = vec![0u8; range.len()];
    f.seek(SeekFrom::Start(range.start as u64))?;
    f.read_exact(&mut dir)?;
    let carry: Vec<(Section, u32)> = parse_directory(&dir, file_len)?
        .into_iter()
        .map(|s| (s, s.kind.flags()))
        .collect();

    f.seek(SeekFrom::End(0))?;
    let (dir_offset, n_sections) = sections.finish(&mut f, &carry)?;
    f.sync_data()?;

    h.dir_offset = dir_offset;
    h.n_sections = n_sections;
    patch(&mut h);
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&h.encode())?;
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

/// Switch the files of a v5 store to TF-IDF normalisation.
///
/// Writes the IDF section and sets the norm kind and the scale in every
/// file's header; no chunk is rewritten, since the norm is derived on read.
/// Call again after the cell set changes to refresh the IDF.
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
pub fn set_tf_idf(paths: &[&str], idf: &[f32], scale: f32) -> Result<(), ScDataErrors> {
    for &p in paths {
        let reader = ParallelSparseReader::new(p)?;
        if reader.file_version() != SC_FILE_VERSION {
            return Err(ScDataErrors::FileVersionMismatch {
                expected: SC_FILE_VERSION,
                found: reader.file_version(),
            });
        }
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
            h.target_size = scale;
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
/// the store's.
pub fn set_cell_ids<S: AsRef<str>>(paths: &[&str], ids: &[S]) -> Result<(), ScDataErrors> {
    for &p in paths {
        let n = ParallelSparseReader::new(p)?.dims().total_cells;
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
/// `Ok(())`, or an error for a file that is not v5.
pub fn set_feature_kind(paths: &[&str], kind: FeatureKind) -> Result<(), ScDataErrors> {
    for &p in paths {
        rewrite_sections(p, SectionWriter::default(), |h| h.feature_kind = kind)?;
    }
    Ok(())
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
#[cfg(test)]
pub(crate) fn check_same_cells(readers: &[&ParallelSparseReader]) -> Result<(), ScDataErrors> {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_kind_byte_and_name_round_trip() {
        for b in 0..=5u8 {
            let k = FeatureKind::from_u8(b).unwrap();
            assert_eq!(k.to_u8(), b);
            assert_eq!(parse_feature_kind(k.name()), Some(k));
        }
        assert!(FeatureKind::from_u8(6).is_err());
    }
}
