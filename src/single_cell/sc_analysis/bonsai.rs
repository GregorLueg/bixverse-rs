//! Bonsai trees over single cell counts.
//!
//! Raw UMI counts come off the gene-major binary file, Sanity turns them into
//! posterior log fold changes with error bars, and Bonsai reconstructs a tree
//! over the cells from those. A layout then puts every node, leaves and
//! inferred ancestors alike, on the plane.
//!
//! Genes are streamed through Sanity in chunks, and each chunk keeps only the
//! genes that would survive Bonsai's ingest (`sanity_gene_passes`). Sanity
//! shares nothing across genes but the per-cell totals, so the chunked run is
//! the same run, and at no point are more than one chunk of counts plus the
//! survivors resident. That is what makes all genes, not only the HVGs, a
//! workable input.
//!
//! Sanity runs on the CPU here. `gpu::sc_gpu::sanity_bonsai_gpu` swaps in the
//! GPU Sanity per chunk; both hand over at [`run_bonsai_sc`], so everything
//! after the posteriors is shared.
//!
//! Sanity is reached through `bonsai_rs::sanity_sc_rs` rather than as a direct
//! dependency, so its `SanityOutput` is always the type
//! `from_sanity_output` expects.

use bonsai_rs::bonsai::{BonsaiParams, bonsai};
use bonsai_rs::errors::BonsaiErrors;
use bonsai_rs::ingest::{from_sanity_output, sanity_gene_passes};
use bonsai_rs::sanity_sc_rs::config::{SanityParams, Verbosity as SanityVerbosity};
use bonsai_rs::sanity_sc_rs::input::CountMatrix;
use bonsai_rs::sanity_sc_rs::{GeneView, SanityOutput, sanity_select};
use bonsai_rs::tree::layout::{Layout, dendrogram, equal_angle, equal_daylight};
use bonsai_rs::tree::{NO_NODE, Tree};
use indexmap::IndexSet;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{RawCounts, SingleCellReading};

///////////////
// Constants //
///////////////

/// Genes read from disk and run through Sanity per chunk.
///
/// Sets the resident input: one chunk of sparse counts (and, on the GPU, its
/// dense rows until the filter has run). At 100k cells and a typical 5 to 10
/// per cent density that is tens of MB of counts, while 1,024 genes still give
/// every Rayon worker hundreds of genes to share out.
pub const SANITY_GENE_CHUNK: usize = 1024;

////////////
// Params //
////////////

/// Which 2D layout to put the tree in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BonsaiLayout {
    /// Equal angle: every subtree gets a wedge proportional to its leaf count.
    #[default]
    EqualAngle,
    /// Equal angle refined towards equal daylight. Returns the equal-angle
    /// layout unrefined above bonsai-rs's node cap (2,048 nodes by default).
    EqualDaylight,
    /// Rectangular dendrogram, leaves along one axis.
    Dendrogram,
}

/// Parse a layout name.
///
/// ### Params
///
/// * `s` - `"equal_angle"`, `"equal_daylight"` or `"dendrogram"`,
///   case-insensitive
///
/// ### Returns
///
/// The layout, or `None` if the name is not recognised.
pub fn parse_bonsai_layout(s: &str) -> Option<BonsaiLayout> {
    match s.to_lowercase().as_str() {
        "equal_angle" => Some(BonsaiLayout::EqualAngle),
        "equal_daylight" => Some(BonsaiLayout::EqualDaylight),
        "dendrogram" => Some(BonsaiLayout::Dendrogram),
        _ => None,
    }
}

/// Parameters for Bonsai over single cell counts.
#[derive(Clone, Copy, Debug)]
pub struct BonsaiScParams {
    /// Sanity run parameters.
    pub sanity: SanityParams,
    /// Bonsai search parameters, feature selection included.
    pub bonsai: BonsaiParams,
    /// Layout of the finished tree.
    pub layout: BonsaiLayout,
    /// Project the layout onto the hyperbolic disk afterwards.
    pub hyperbolic: bool,
}

impl Default for BonsaiScParams {
    /// Upstream defaults, with two changes for a tree meant to be drawn: the
    /// tree is rerooted for display, and the per-node posteriors are skipped
    /// (they are `2 * n_nodes * n_genes` values nobody here reads).
    ///
    /// ### Returns
    ///
    /// The default parameter set.
    fn default() -> Self {
        Self {
            sanity: SanityParams::default(),
            bonsai: BonsaiParams {
                reroot: true,
                skip_posteriors: true,
                ..BonsaiParams::default()
            },
            layout: BonsaiLayout::default(),
            hyperbolic: false,
        }
    }
}

/////////////
// Results //
/////////////

/// A Bonsai tree with its layout.
pub struct BonsaiScResult {
    /// Parent of each node, [`NO_NODE`] for the root. Nodes `0..n_cells` are
    /// the selected cells in selection order; the rest are inferred ancestors.
    pub parent: Vec<u32>,
    /// Length of the branch above each node. The root's entry is unused.
    pub branch: Vec<f64>,
    /// Horizontal coordinate of each node.
    pub x: Vec<f64>,
    /// Vertical coordinate of each node.
    pub y: Vec<f64>,
    /// Number of leaves, which is the number of selected cells.
    pub n_leaves: usize,
    /// Final tree loglikelihood, up to an additive constant.
    pub loglik: f64,
    /// Name and loglikelihood after each search step.
    pub steps: Vec<(String, f64)>,
    /// Input gene indices the tree was built on, ascending.
    pub genes_used: Vec<usize>,
}

//////////////////
// Sanity input //
//////////////////

/// Read one chunk of genes as a Sanity count matrix.
///
/// Genes with no counts in the selected cells are left out, since Sanity
/// cannot fit them.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `genes` - Genes to read, 0-indexed
/// * `cell_set` - Cells to keep, in leaf order
///
/// ### Returns
///
/// The count matrix, cells renumbered `0..cell_set.len()`, and the input gene
/// index of each of its columns; `None` if every gene in the chunk was empty.
pub fn read_count_chunk<G: SingleCellReading>(
    gene_reader: &G,
    genes: &[usize],
    cell_set: &IndexSet<u32>,
) -> Result<Option<(CountMatrix, Vec<usize>)>, BixverseErrors> {
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut indptr: Vec<usize> = vec![0];
    let mut kept: Vec<usize> = Vec::with_capacity(genes.len());

    for (&gene, chunk) in genes
        .iter()
        .zip(gene_reader.read_gene_parallel_filtered(genes, cell_set)?)
    {
        if chunk.indices.is_empty() {
            continue;
        }
        // filtering renumbers cells by their position in `cell_set`, in
        // ascending order, which is the layout `CountMatrix` requires
        indices.extend_from_slice(&chunk.indices);
        match chunk.data_raw {
            RawCounts::U16(v) => values.extend(v.into_iter().map(u32::from)),
            RawCounts::U32(v) => values.extend(v),
        }
        indptr.push(indices.len());
        kept.push(gene);
    }

    if kept.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        CountMatrix::new(indices, values, indptr, cell_set.len())?,
        kept,
    )))
}

/// Stream genes through Sanity in chunks and keep what each chunk returns.
///
/// Library sizes come once from the cell-major file's chunk headers: the total
/// over all genes, which is what Sanity asks for, and the only quantity its
/// model shares across genes. So each chunk is fitted exactly as it would be in
/// one run over every gene. `run_chunk` decides what a chunk keeps, normally
/// through `sanity_select` or `sanity_gpu_select` with [`keep_for_bonsai`].
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to run, 0-indexed
/// * `gene_chunk` - Genes per chunk, usually [`SANITY_GENE_CHUNK`]
/// * `verbosity` - How much to print
/// * `run_chunk` - Sanity over one chunk's counts and the cell totals
///
/// ### Returns
///
/// The kept genes of every chunk as one Sanity output, rows in input order,
/// with `genes` holding their input gene indices.
pub fn stream_sanity<G, C, F>(
    gene_reader: &G,
    cell_reader: &C,
    cell_indices: &[usize],
    gene_indices: &[usize],
    gene_chunk: usize,
    verbosity: Verbosity,
    mut run_chunk: F,
) -> Result<SanityOutput<f32>, BixverseErrors>
where
    G: SingleCellReading,
    C: SingleCellReading,
    F: FnMut(&CountMatrix, &[f64]) -> Result<SanityOutput<f32>, BixverseErrors>,
{
    let started = Instant::now();
    let n_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&c| c as u32).collect();
    let cell_totals: Vec<f64> = cell_reader
        .read_cell_library_sizes(cell_indices)?
        .into_iter()
        .map(|t| t as f64)
        .collect();

    let mut out = SanityOutput {
        log_fold_changes: Vec::new(),
        error_bars: Vec::new(),
        mean_log_quotient: Vec::new(),
        mean_log_quotient_error: Vec::new(),
        variance: Vec::new(),
        genes: Vec::new(),
        n_genes: 0,
        n_cells,
    };

    let n_chunks = gene_indices.len().div_ceil(gene_chunk.max(1));
    for (i, genes) in gene_indices.chunks(gene_chunk.max(1)).enumerate() {
        let Some((counts, chunk_genes)) = read_count_chunk(gene_reader, genes, &cell_set)? else {
            continue;
        };
        let post = run_chunk(&counts, &cell_totals)?;
        drop(counts);

        out.log_fold_changes.extend(post.log_fold_changes);
        out.error_bars.extend(post.error_bars);
        out.mean_log_quotient.extend(post.mean_log_quotient);
        out.mean_log_quotient_error
            .extend(post.mean_log_quotient_error);
        out.variance.extend(post.variance);
        // `post.genes` are positions in this chunk's count matrix
        out.genes.extend(post.genes.iter().map(|&g| chunk_genes[g]));

        if verbosity.normal_verbosity() {
            println!(
                "Sanity chunk {}/{}: kept {} of {} genes ({} kept so far, {:.2?}).",
                i + 1,
                n_chunks,
                post.n_genes,
                genes.len(),
                out.genes.len(),
                started.elapsed()
            );
        }
    }

    out.n_genes = out.genes.len();
    Ok(out)
}

/// The Sanity keep test that matches Bonsai's ingest.
///
/// ### Params
///
/// * `params` - Parameters carrying the ingest knobs
///
/// ### Returns
///
/// A predicate for `sanity_select` and `sanity_gpu_select` that keeps a gene
/// only if `from_sanity` and `prepare` would.
pub fn keep_for_bonsai(params: &BonsaiScParams) -> impl Fn(GeneView<'_>) -> bool + Sync + use<> {
    let ingest = params.bonsai.ingest;
    move |g| sanity_gene_passes(g.log_fold_changes, g.error_bars, g.variance, &ingest)
}

/// The Sanity parameters for one chunk.
///
/// Silent, since [`stream_sanity`] reports per chunk and Sanity's own header
/// would repeat for every one of them.
///
/// ### Params
///
/// * `params` - Parameters carrying the Sanity set
///
/// ### Returns
///
/// `params.sanity` with its verbosity off.
pub(crate) fn chunk_sanity_params(params: &BonsaiScParams) -> SanityParams {
    SanityParams {
        verbosity: SanityVerbosity::Quiet,
        ..params.sanity
    }
}

////////////
// Layout //
////////////

/// Lay a tree out on the plane.
///
/// ### Params
///
/// * `tree` - The tree
/// * `layout` - Which layout
/// * `hyperbolic` - Project onto the hyperbolic disk afterwards
///
/// ### Returns
///
/// Node coordinates, indexed like the tree's nodes.
fn layout_tree(
    tree: &Tree,
    layout: BonsaiLayout,
    hyperbolic: bool,
) -> Result<Layout, BixverseErrors> {
    let out = match layout {
        BonsaiLayout::EqualAngle => equal_angle(tree, None)?,
        BonsaiLayout::EqualDaylight => equal_daylight(tree, None)?,
        BonsaiLayout::Dendrogram => dendrogram(tree, None)?,
    };
    Ok(if hyperbolic {
        out.hyperbolic(None)
    } else {
        out
    })
}

/// Lay out a tree given as parent and branch arrays.
///
/// For redrawing a finished tree without searching again. The arena relabels
/// internal nodes into level order, so the returned parent and branch arrays
/// can number the ancestors differently from the input; leaves keep
/// `0..n_leaves`. Callers should replace their node table with the output
/// rather than attach the coordinates to the input.
///
/// ### Params
///
/// * `parent` - Parent per node, [`NO_NODE`] for the root
/// * `branch` - Branch length above each node
/// * `n_leaves` - Number of leaves, which occupy `0..n_leaves`
/// * `layout` - Which layout
/// * `hyperbolic` - Project onto the hyperbolic disk afterwards
///
/// ### Returns
///
/// The tree's parent array, branch lengths and the `x` and `y` coordinates,
/// all indexed by the tree's own node numbering.
pub fn bonsai_layout(
    parent: Vec<u32>,
    branch: Vec<f64>,
    n_leaves: usize,
    layout: BonsaiLayout,
    hyperbolic: bool,
) -> Result<(Vec<u32>, Vec<f64>, Layout), BixverseErrors> {
    let tree = Tree::from_parents(parent, branch, n_leaves)?;
    let coords = layout_tree(&tree, layout, hyperbolic)?;
    let (parent, branch) = tree_arrays(&tree);
    Ok((parent, branch, coords))
}

/// Flatten a tree into its parent and branch arrays.
///
/// ### Params
///
/// * `tree` - The tree
///
/// ### Returns
///
/// Parent per node ([`NO_NODE`] for the root) and branch length per node.
fn tree_arrays(tree: &Tree) -> (Vec<u32>, Vec<f64>) {
    let parent = (0..tree.n_nodes() as u32)
        .map(|v| tree.parent(v).unwrap_or(NO_NODE))
        .collect();
    (parent, tree.branches().to_vec())
}

////////////
// Bonsai //
////////////

/// Build and lay out a Bonsai tree from a Sanity run.
///
/// Takes the Sanity output by value and drops it once the ingest has copied
/// what it needs, so the gene-major posteriors and the cell-major search input
/// are not both held for the whole search.
///
/// ### Params
///
/// * `post` - Sanity posteriors, `genes` holding input gene indices as
///   [`stream_sanity`] leaves them
/// * `n_candidates` - Genes that went into Sanity, for the error when none
///   survived
/// * `params` - Bonsai and layout parameters; `params.sanity` is not read
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The tree, its layout, and the genes it was built on.
pub fn run_bonsai_sc(
    post: SanityOutput<f32>,
    n_candidates: usize,
    params: &BonsaiScParams,
    verbosity: Verbosity,
) -> Result<BonsaiScResult, BixverseErrors> {
    if post.n_genes == 0 {
        return Err(BonsaiErrors::NoFeaturesRetained {
            n_features: n_candidates,
            threshold: params.bonsai.ingest.min_signal_to_noise,
        }
        .into());
    }
    let lik = from_sanity_output(&post, Some(params.bonsai.ingest))?;
    drop(post);

    if verbosity.normal_verbosity() {
        println!(
            "Sanity ingest kept {} genes, dropped {} as ill-conditioned.",
            lik.features.len(),
            lik.dropped.len()
        );
    }

    let out = bonsai(
        &lik.means,
        &lik.sds,
        lik.n_cells,
        lik.features.len(),
        Some(&lik.variances),
        Some(params.bonsai),
        bonsai_verbosity(verbosity),
    )?;

    let coords = layout_tree(&out.tree, params.layout, params.hyperbolic)?;
    let (parent, branch) = tree_arrays(&out.tree);

    Ok(BonsaiScResult {
        parent,
        branch,
        x: coords.x,
        y: coords.y,
        n_leaves: out.tree.n_leaves(),
        loglik: out.loglik,
        steps: out
            .steps
            .iter()
            .map(|s| (s.step.to_string(), s.loglik))
            .collect(),
        // bonsai's features index the ingest's, which are input gene indices
        genes_used: out.features.iter().map(|&f| lik.features[f]).collect(),
    })
}

/// Counts on disk to a laid-out Bonsai tree, Sanity on the CPU.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to consider, 0-indexed. All expressed genes is
///   fine: only the ones passing Bonsai's ingest filters are kept
/// * `params` - Sanity, Bonsai and layout parameters
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The tree, its layout, and the genes it was built on.
pub fn sanity_bonsai_sc<G, C>(
    gene_reader: &G,
    cell_reader: &C,
    cell_indices: &[usize],
    gene_indices: &[usize],
    params: &BonsaiScParams,
    verbosity: Verbosity,
) -> Result<BonsaiScResult, BixverseErrors>
where
    G: SingleCellReading,
    C: SingleCellReading,
{
    let sanity_params = chunk_sanity_params(params);
    let keep = keep_for_bonsai(params);
    let post = stream_sanity(
        gene_reader,
        cell_reader,
        cell_indices,
        gene_indices,
        SANITY_GENE_CHUNK,
        verbosity,
        |counts, totals| Ok(sanity_select(counts, totals, Some(sanity_params), &keep)?),
    )?;
    run_bonsai_sc(post, gene_indices.len(), params, verbosity)
}

/// This crate's verbosity as bonsai-rs's.
///
/// ### Params
///
/// * `verbosity` - This crate's verbosity
///
/// ### Returns
///
/// The matching bonsai-rs level.
fn bonsai_verbosity(verbosity: Verbosity) -> bonsai_rs::utils::verbosity::Verbosity {
    use bonsai_rs::utils::verbosity::Verbosity as BonsaiVerbosity;
    match verbosity {
        Verbosity::Quiet => BonsaiVerbosity::Quiet,
        Verbosity::Normal => BonsaiVerbosity::Normal,
        Verbosity::Detailed => BonsaiVerbosity::Detailed,
    }
}
