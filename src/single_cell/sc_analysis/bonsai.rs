//! Bonsai trees over single cell counts.
//!
//! Raw UMI counts come off the gene-major binary file, Sanity turns them into
//! posterior log fold changes with error bars, and Bonsai reconstructs a tree
//! over the cells from those. A layout then puts every node, leaves and
//! inferred ancestors alike, on the plane.
//!
//! Sanity runs on the CPU here. `gpu::sc_gpu::sanity_bonsai_gpu` swaps in the
//! GPU Sanity; both hand over at [`run_bonsai_sc`], so everything after the
//! posteriors is shared.
//!
//! Sanity is reached through `bonsai_rs::sanity_sc_rs` rather than as a direct
//! dependency, so its `SanityOutput` is always the type
//! `from_sanity_output` expects.

use bonsai_rs::bonsai::{BonsaiParams, bonsai};
use bonsai_rs::ingest::from_sanity_output;
use bonsai_rs::sanity_sc_rs::config::SanityParams;
use bonsai_rs::sanity_sc_rs::input::CountMatrix;
use bonsai_rs::sanity_sc_rs::{SanityOutput, sanity};
use bonsai_rs::tree::layout::{Layout, dendrogram, equal_angle, equal_daylight};
use bonsai_rs::tree::{NO_NODE, Tree};
use indexmap::IndexSet;
use std::time::Instant;

use crate::prelude::*;
use crate::single_cell::sc_data::data_io::{RawCounts, SingleCellReading};

///////////////
// Constants //
///////////////

/// Genes read from disk per batch while assembling the Sanity input.
///
/// Each chunk also carries the f16 normalised values Sanity never reads, so
/// batching bounds that transient copy rather than the count matrix itself,
/// which has to be whole.
const SANITY_READ_BATCH: usize = 1024;

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

/// Sanity input assembled from the binary file.
pub struct SanityCounts {
    /// Raw counts, gene-major, cells renumbered `0..n_cells` in the order of
    /// the cell selection.
    pub counts: CountMatrix,
    /// Total UMI count of every selected cell over all genes.
    pub cell_totals: Vec<f64>,
    /// Input gene index of each column of `counts`. Genes with no counts in the
    /// selected cells are left out, since Sanity cannot fit them.
    pub genes: Vec<usize>,
}

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

/// Read the raw counts of the selected genes and cells for Sanity.
///
/// Library sizes come from the cell-major file's chunk headers, which is the
/// total over all genes that Sanity asks for. Genes are read in batches of
/// [`SANITY_READ_BATCH`] and only their raw counts and cell indices kept.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to read, 0-indexed
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The count matrix, the per-cell totals, and the gene index of each column.
pub fn sanity_counts<G, C>(
    gene_reader: &G,
    cell_reader: &C,
    cell_indices: &[usize],
    gene_indices: &[usize],
    verbosity: Verbosity,
) -> Result<SanityCounts, BixverseErrors>
where
    G: SingleCellReading,
    C: SingleCellReading,
{
    let started = Instant::now();
    let n_cells = cell_indices.len();
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&c| c as u32).collect();

    let cell_totals: Vec<f64> = cell_reader
        .read_cell_library_sizes(cell_indices)?
        .into_iter()
        .map(|t| t as f64)
        .collect();

    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut indptr: Vec<usize> = vec![0];
    let mut genes: Vec<usize> = Vec::with_capacity(gene_indices.len());

    for batch in gene_indices.chunks(SANITY_READ_BATCH) {
        let chunks = gene_reader.read_gene_parallel_filtered(batch, &cell_set)?;
        for (&gene, chunk) in batch.iter().zip(chunks) {
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
            genes.push(gene);
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Read {} of {} genes ({} with no counts in the selection) over {} cells in {:.2?}.",
            genes.len(),
            gene_indices.len(),
            gene_indices.len() - genes.len(),
            n_cells,
            started.elapsed()
        );
    }

    Ok(SanityCounts {
        counts: CountMatrix::new(indices, values, indptr, n_cells)?,
        cell_totals,
        genes,
    })
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
/// * `post` - Sanity posteriors over the columns of `genes`
/// * `genes` - Input gene index of each Sanity gene, from [`SanityCounts`]
/// * `params` - Bonsai and layout parameters; `params.sanity` is not read
/// * `verbosity` - How much to print
///
/// ### Returns
///
/// The tree, its layout, and the genes it was built on.
pub fn run_bonsai_sc(
    post: SanityOutput<f32>,
    genes: &[usize],
    params: &BonsaiScParams,
    verbosity: Verbosity,
) -> Result<BonsaiScResult, BixverseErrors> {
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
        // bonsai's features index the ingest's, which index Sanity's genes
        genes_used: out
            .features
            .iter()
            .map(|&f| genes[lik.features[f]])
            .collect(),
    })
}

/// Counts on disk to a laid-out Bonsai tree, Sanity on the CPU.
///
/// ### Params
///
/// * `gene_reader` - Reader over the gene-major file
/// * `cell_reader` - Reader over the cell-major file
/// * `cell_indices` - Cells to keep, 0-indexed. Sets the leaf order
/// * `gene_indices` - Genes to use, 0-indexed. Highly variable genes, usually
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
    let input = sanity_counts(
        gene_reader,
        cell_reader,
        cell_indices,
        gene_indices,
        verbosity,
    )?;
    let post = sanity::<f32>(
        &input.counts,
        &input.cell_totals,
        Some(sanity_params(params, verbosity)),
    )?;
    drop(input.counts);
    run_bonsai_sc(post, &input.genes, params, verbosity)
}

/// The Sanity parameters with this crate's verbosity applied.
///
/// ### Params
///
/// * `params` - Parameters carrying the Sanity set
/// * `verbosity` - This crate's verbosity
///
/// ### Returns
///
/// `params.sanity` with its verbosity replaced.
pub(crate) fn sanity_params(params: &BonsaiScParams, verbosity: Verbosity) -> SanityParams {
    use bonsai_rs::sanity_sc_rs::config::Verbosity as SanityVerbosity;
    SanityParams {
        verbosity: match verbosity {
            Verbosity::Quiet => SanityVerbosity::Quiet,
            Verbosity::Normal => SanityVerbosity::Normal,
            Verbosity::Detailed => SanityVerbosity::Detailed,
        },
        ..params.sanity
    }
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
