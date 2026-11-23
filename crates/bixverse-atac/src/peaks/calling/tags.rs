//! Tags from the fragment file: read, sort, deduplicate, spill to scratch.

use super::*;

/////////////
// Reading //
/////////////

/// One chromosome's tags of one worker, per group.
#[derive(Clone)]
pub(super) struct GroupTags {
    /// Tags per group.
    pub(super) tags: Vec<Vec<u32>>,
    /// First fragments as `(start, length)` per group; only filled when the
    /// tag size is needed.
    pub(super) first: Vec<Vec<(u32, u32)>>,
}

impl GroupTags {
    /// No tags yet.
    ///
    /// ### Params
    ///
    /// * `n_groups` - Number of groups
    ///
    /// ### Returns
    ///
    /// Empty lists for every group.
    pub(super) fn new(n_groups: usize) -> Self {
        Self {
            tags: vec![Vec::new(); n_groups],
            first: vec![Vec::new(); n_groups],
        }
    }

    /// Add another worker's tags of the same chromosome.
    ///
    /// ### Params
    ///
    /// * `other` - The other worker's tags
    pub(super) fn merge(&mut self, other: GroupTags) {
        for (a, b) in self.tags.iter_mut().zip(other.tags) {
            extend_larger(a, b);
        }
        for (a, b) in self.first.iter_mut().zip(other.first) {
            a.extend(b);
            keep_first_fragments(a);
        }
    }
}

/// Keep the [`TAG_SIZE_SAMPLE`] smallest `(start, length)` pairs.
///
/// ### Params
///
/// * `first` - The pairs
pub(super) fn keep_first_fragments(first: &mut Vec<(u32, u32)>) {
    first.sort_unstable();
    first.truncate(TAG_SIZE_SAMPLE);
}

/// Read the tags of every group from one batch or the whole file.
///
/// ### Params
///
/// * `path` - Fragment file
/// * `batch` - Chromosomes to read, `None` for the whole file
/// * `index` - Cells
/// * `groups` - Group of each cell
/// * `n_groups` - Number of groups
/// * `source` - What becomes a tag
/// * `track_first` - Whether to keep the first fragments, for the tag size
///
/// ### Returns
///
/// The unsorted tags as `(group, chromosome, tags)` and the first fragments
/// per group, or an I/O or parse error.
pub(super) fn read_tags(
    path: &Path,
    batch: &FragmentBatch,
    index: &BarcodeIndex,
    groups: &[u32],
    n_groups: usize,
    source: TagSource,
    track_first: bool,
) -> Result<(Vec<TagJob>, FirstFragments), AtacErrors> {
    let fold = |acc: &mut ChromAcc<GroupTags>, chrom: &str, frags: &[Fragment]| {
        let chrom_tags = acc.entry(chrom, || GroupTags::new(n_groups));
        let tags = &mut chrom_tags.tags;
        for f in frags {
            let g = groups[f.cell as usize] as usize;
            tags[g].push(f.start);
            if source == TagSource::Insertion {
                tags[g].push(f.end);
            }
        }
        if track_first {
            let first = &mut chrom_tags.first;
            for f in frags {
                // runs reach a worker in any order, so keep the smallest
                // starts seen, not the first ones
                let fs = &mut first[groups[f.cell as usize] as usize];
                fs.push((f.start, f.len()));
                if fs.len() > 2 * TAG_SIZE_SAMPLE {
                    keep_first_fragments(fs);
                }
            }
        }
    };
    let reduce = |mut a: ChromAcc<GroupTags>, b: ChromAcc<GroupTags>| {
        a.merge(b, GroupTags::merge);
        a
    };
    let acc = fold_batch(path, batch, index, ChromAcc::default, fold, reduce)?;

    let mut firsts: FirstFragments = vec![FxHashMap::default(); n_groups];
    let mut jobs = Vec::new();
    for (chrom, GroupTags { tags, first }) in acc.chroms.into_iter().zip(acc.values) {
        for (g, mut f) in first.into_iter().enumerate() {
            if !f.is_empty() {
                keep_first_fragments(&mut f);
                firsts[g].insert(chrom.clone(), f);
            }
        }
        for (g, t) in tags.into_iter().enumerate() {
            if !t.is_empty() {
                jobs.push((g, chrom.clone(), t));
            }
        }
    }
    Ok((jobs, firsts))
}

/// Sort tags and drop duplicates above `keep`.
///
/// ### Params
///
/// * `tags` - Tag positions, any order
/// * `keep` - Duplicate handling
pub(super) fn sort_and_dedup(tags: &mut Vec<u32>, keep: KeepDup) {
    tags.par_sort_unstable();
    let KeepDup::AtMost(max) = keep else {
        return;
    };
    let mut w = 0;
    let mut run = 0u32;
    for r in 0..tags.len() {
        run = if r > 0 && tags[r] == tags[r - 1] {
            run + 1
        } else {
            1
        };
        if run <= max {
            tags[w] = tags[r];
            w += 1;
        }
    }
    tags.truncate(w);
}

/// MACS's tag size of a fragment start group: the truncated mean length of
/// its first fragments.
///
/// MACS takes the first lines of the file it reads, which for a group is its
/// first fragments on the first chromosome, in file order, that it has any
/// on. Without an index only the file's first chromosome is known; past it
/// the fallback is name order.
///
/// ### Params
///
/// * `first` - The group's first fragments per chromosome
/// * `order` - Chromosomes in file order, as far as known
///
/// ### Returns
///
/// The tag size, 0 for an empty group.
pub(super) fn fragment_tag_size(
    first: &FxHashMap<String, Vec<(u32, u32)>>,
    order: &[String],
) -> u32 {
    let chrom = order
        .iter()
        .find(|c| first.contains_key(*c))
        .or_else(|| first.keys().min());
    let Some(chrom) = chrom else {
        return 0;
    };
    let mut frags = first[chrom].clone();
    frags.sort_unstable();
    frags.truncate(TAG_SIZE_SAMPLE);
    let sum: u64 = frags.iter().map(|&(_, l)| l as u64).sum();
    (sum as f64 / frags.len() as f64) as u32
}

///////////
// Spill //
///////////

/// Where one group's tags of one chromosome sit in the spill.
pub(super) struct SpillEntry {
    /// Group.
    pub(super) group: usize,
    /// Chromosome.
    pub(super) chrom: String,
    /// Byte offset.
    pub(super) offset: u64,
    /// Number of tags.
    pub(super) len: usize,
}

/// Sorted, deduplicated tags kept on disk between passes, in native byte
/// order. The file is removed on drop.
pub(super) struct TagSpill {
    /// The scratch file.
    pub(super) path: PathBuf,
    /// Open for appending and mapping.
    pub(super) file: File,
    /// Its contents, in write order.
    pub(super) entries: Vec<SpillEntry>,
    /// Bytes written.
    pub(super) size: u64,
}

impl TagSpill {
    /// Create an empty spill.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory of the scratch file
    ///
    /// ### Returns
    ///
    /// The spill, or an I/O error.
    pub(super) fn create(dir: &Path) -> Result<Self, AtacErrors> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("bv_peak_tags_{}_{nanos}.bin", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            entries: Vec::new(),
            size: 0,
        })
    }

    /// Append one group's tags of one chromosome.
    ///
    /// ### Params
    ///
    /// * `group` - Group
    /// * `chrom` - Chromosome
    /// * `tags` - Sorted tags, not empty
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an I/O error.
    pub(super) fn push(
        &mut self,
        group: usize,
        chrom: String,
        tags: &[u32],
    ) -> Result<(), AtacErrors> {
        // SAFETY: u32 has no padding, so its bytes are plain initialised
        // memory of the slice's length times 4.
        let bytes =
            unsafe { std::slice::from_raw_parts(tags.as_ptr().cast::<u8>(), tags.len() * 4) };
        self.file.write_all(bytes)?;
        self.entries.push(SpillEntry {
            group,
            chrom,
            offset: self.size,
            len: tags.len(),
        });
        self.size += bytes.len() as u64;
        Ok(())
    }

    /// Map one entry's tags.
    ///
    /// ### Params
    ///
    /// * `e` - The entry
    ///
    /// ### Returns
    ///
    /// The mapping, or an I/O error.
    pub(super) fn map(&self, e: &SpillEntry) -> Result<Mmap, AtacErrors> {
        // SAFETY: the scratch file is private to this spill and not written
        // while mapped.
        let map = unsafe {
            MmapOptions::new()
                .offset(e.offset)
                .len(e.len * 4)
                .map(&self.file)?
        };
        Ok(map)
    }
}

impl Drop for TagSpill {
    /// Remove the scratch file.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A mapped spill entry as tags.
///
/// ### Params
///
/// * `map` - From [`TagSpill::map`]
///
/// ### Returns
///
/// The tags.
pub(super) fn mapped_tags(map: &Mmap) -> &[u32] {
    // SAFETY: every bit pattern is a valid u32, and the entries start at
    // multiples of 4 bytes in the file, so the mapping is 4-byte aligned;
    // align_to checks that.
    let (pre, tags, post) = unsafe { map.align_to::<u32>() };
    assert!(
        pre.is_empty() && post.is_empty(),
        "spill mapping is not aligned"
    );
    tags
}
