//! Errors raised by `bixverse-sp`.

use thiserror::Error;

/// Error variants raised by `bixverse-sp`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum SpErrors {
    // -- lower crates --
    /// Propagates an error from `bixverse-core`.
    #[error(transparent)]
    Core(#[from] bixverse_core::errors::CoreErrors),

    /// Propagates an error from `bixverse-methods`.
    #[error(transparent)]
    Methods(#[from] bixverse_methods::errors::MethodsErrors),

    /// Propagates an error from `bixverse-sc-data`.
    #[error(transparent)]
    ScData(#[from] bixverse_sc_data::errors::ScDataErrors),

    /// Propagates an error from `bixverse-sc`.
    #[error(transparent)]
    Sc(#[from] bixverse_sc::errors::ScErrors),

    // -- hdf5 --
    /// Propagates an error from the HDF5 library.
    #[cfg(feature = "h5")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    // -- spatial --
    /// Spatial graph node count does not match the per-sample spot count.
    #[error("Spatial: graph has {graph_nodes} nodes but {n_spots} spots were provided")]
    SpatialGraphSpotMismatch {
        /// Number of nodes in the per-sample spatial graph.
        graph_nodes: usize,
        /// Length of `spots_to_keep` (per-sample spot count).
        n_spots: usize,
    },

    /// Per-sample coordinate count does not match the per-sample spot count.
    #[error("Spatial: got {n_coords} coordinates but {n_spots} spots were provided")]
    SpatialCoordSpotMismatch {
        /// Number of coordinate pairs supplied.
        n_coords: usize,
        /// Length of `spots_to_keep`.
        n_spots: usize,
    },

    /// Neighbourhood enrichment labels do not match the graph node count.
    #[error("Spatial: labels length {n_labels} does not match graph node count {n_nodes}")]
    NhoodLabelLengthMismatch {
        /// Length of the labels slice.
        n_labels: usize,
        /// Number of graph nodes.
        n_nodes: usize,
    },

    /// Encountered a label code outside `0..label_levels.len()`.
    #[error("Spatial: label code {code} >= number of levels {n_levels}")]
    NhoodLabelOutOfRange {
        /// Offending code.
        code: u32,
        /// Number of label levels declared.
        n_levels: usize,
    },

    /// `n_permutations == 0` for neighbourhood enrichment.
    #[error("Spatial: n_permutations must be >= 1")]
    NhoodZeroPermutations,

    /// Lattice index arrays do not match the coordinate count.
    #[error(
        "Spatial: lattice arrays have lengths {n_array_row} (row) and {n_array_col} (col) but {n_spots} spots were provided"
    )]
    SpatialLatticeLengthMismatch {
        /// Length of `array_row`.
        n_array_row: usize,
        /// Length of `array_col`.
        n_array_col: usize,
        /// Number of coordinate pairs supplied.
        n_spots: usize,
    },

    /// Two spots share the same lattice coordinate, which makes the lookup
    /// driving the exact lattice adjacency ambiguous.
    #[error("Spatial: lattice coordinate ({array_row}, {array_col}) occurs more than once")]
    SpatialLatticeDuplicateCoord {
        /// Offending `array_row`.
        array_row: i32,
        /// Offending `array_col`.
        array_col: i32,
    },

    /// A spot coordinate is not finite.
    ///
    /// `NA_real_` in the R coordinate matrix, or an infinity out of a bad
    /// rescale. Rejected at the boundary because nothing downstream notices:
    /// the tile window saturates to the image origin, the radius graph buckets
    /// the spot at `(0, 0)` and the SPARK-X bandwidth quantiles come back
    /// short.
    #[error("Spatial: the coordinate at index {index} is not finite")]
    SpatialNonFiniteCoord {
        /// Position of the offending coordinate in the input slice.
        index: usize,
    },

    // -- spatial ingest --
    /// The h5ad carries no `obsm/spatial` array.
    ///
    /// A bare `filtered_feature_bc_matrix.h5` hits this: the coordinates live
    /// in a sibling `spatial/` directory rather than in the file, so there is
    /// nothing to read and the answer is to go and ask for that directory.
    #[cfg(feature = "h5")]
    #[error(
        "Spatial: '{path}' not found. A file without it carries no coordinates; if this came from Space Ranger, load the run directory with load_visium() instead"
    )]
    SpatialObsmMissing {
        /// The path that was looked for.
        path: String,
    },

    /// `obsm/spatial` is not an N x 2 (or wider) array.
    #[cfg(feature = "h5")]
    #[error("Spatial: '{path}' has shape {shape}, expected two dimensions with at least 2 columns")]
    SpatialObsmShape {
        /// The offending dataset.
        path: String,
        /// Its shape, as reported by HDF5.
        shape: String,
    },

    /// A numeric HDF5 dataset is stored as something that is not a number.
    ///
    /// `obsm/spatial` is `int64` in every survey file and the scale factors mix
    /// `int64` and `float64` inside one group, so the reader dispatches on the
    /// descriptor rather than assuming. This is what is left over.
    #[cfg(feature = "h5")]
    #[error("Spatial: dataset '{path}' has non-numeric type {dtype}")]
    H5UnexpectedNumericType {
        /// Path of the dataset.
        path: String,
        /// The type descriptor found.
        dtype: String,
    },

    /// The requested `uns/spatial` library is not in the file.
    #[cfg(feature = "h5")]
    #[error("Spatial: uns/spatial library '{library_id}' not found. Available: {available}")]
    SpatialLibraryNotFound {
        /// The library that was asked for.
        library_id: String,
        /// What the file actually holds.
        available: String,
    },

    /// The file carries several `uns/spatial` libraries and none was named.
    ///
    /// Each library has its own scale factors and its own images, so picking
    /// one arbitrarily would attach the wrong ones to the coordinates.
    #[cfg(feature = "h5")]
    #[error(
        "Spatial: uns/spatial holds several libraries and none was requested. Available: {available}"
    )]
    SpatialLibraryAmbiguous {
        /// What the file holds.
        available: String,
    },

    /// `spots_to_keep` repeats a global spot index.
    ///
    /// The local index space is defined by position in the deduplicating
    /// `IndexSet` the readers filter against, while `n_spots` is the raw slice
    /// length. A repeat puts the two out of step without going out of bounds,
    /// so expression pairs with the wrong spot's coordinates and graph row
    /// from the first duplicate onwards.
    #[error("Spatial: spot index {spot} occurs more than once in spots_to_keep (at index {index})")]
    SpatialDuplicateSpot {
        /// The repeated global spot index.
        spot: usize,
        /// Position of the repeat in `spots_to_keep`.
        index: usize,
    },

    /// A node's neighbour list and weight list have different lengths.
    #[error("Spatial: node {node} has {n_neighbours} neighbours but {n_weights} weights")]
    SpatialAdjacencyRagged {
        /// Offending node.
        node: usize,
        /// Length of `neighbours[node]`.
        n_neighbours: usize,
        /// Length of `weights[node]`.
        n_weights: usize,
    },

    /// A neighbour index falls outside the local `0..n_nodes` space.
    ///
    /// A 1-based neighbour list out of R is the usual cause.
    #[error("Spatial: node {node} lists neighbour {neighbour}, which is outside 0..{n_nodes}")]
    SpatialNeighbourOutOfRange {
        /// Offending node.
        node: usize,
        /// The out-of-range neighbour index.
        neighbour: usize,
        /// Number of nodes in the local index space.
        n_nodes: usize,
    },

    /// The automatic Gaussian bandwidth could not be derived because the graph
    /// carries no edge of positive length.
    #[error(
        "Spatial: cannot resolve the Gaussian bandwidth, the graph has no edge with a positive length"
    )]
    SpatialGraphBandwidthUnresolved,

    // -- spatial / image --
    /// Wraps any error out of the `image` crate while decoding a whole image.
    ///
    /// Covers unreadable files, unsupported codecs (only PNG, JPEG and TIFF are
    /// compiled in) and truncated data.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: decoding failed: {0}")]
    ImageDecode(#[from] image::ImageError),

    /// Wraps any error out of `openslide-rs`.
    ///
    /// Covers slide opening, level queries and windowed region reads. The
    /// wrapped error carries the original OpenSlide message.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: OpenSlide error: {0}")]
    OpenSlide(#[from] openslide_rs::errors::OpenSlideError),

    /// The file could not be opened by either backing implementation.
    ///
    /// Raised by the probe in [`crate::sp_image::source::open_image_source`] when
    /// OpenSlide does not recognise the file and the `image` crate cannot
    /// decode it either.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: '{path}' is readable by neither OpenSlide nor the image decoder")]
    ImageSourceUnrecognised {
        /// Path that was probed.
        path: String,
    },

    /// A requested window lies wholly outside the backing image.
    ///
    /// Windows that only partly overlap get clamped and come back smaller.
    /// This fires only when the intersection with the image is empty.
    #[cfg(feature = "spatial-image")]
    #[error(
        "Spatial image: window at ({x}, {y}) size {width}x{height} does not intersect the {img_width}x{img_height} image"
    )]
    ImageWindowOutOfBounds {
        /// Window origin, x.
        x: u32,
        /// Window origin, y.
        y: u32,
        /// Window width.
        width: u32,
        /// Window height.
        height: u32,
        /// Backing image width.
        img_width: u32,
        /// Backing image height.
        img_height: u32,
    },

    /// The slide reports no pyramid levels, so nothing can be read from it.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: the slide reports zero pyramid levels")]
    SlideNoLevels,

    /// A stain matrix could not be inverted.
    ///
    /// Means the three stain vectors are linearly dependent, so the colour
    /// space they span is degenerate and deconvolution has no unique solution.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: the stain matrix is singular and cannot be inverted")]
    StainMatrixSingular,

    /// `glcm_levels` outside the usable range.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: glcm_levels must be in 2..=255, got {0}")]
    GlcmLevelsOutOfRange(u8),

    /// No GLCM offsets were supplied, so no co-occurrence can be counted.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: at least one GLCM offset is required")]
    GlcmNoOffsets,

    /// A quantised value fell outside `0..levels`.
    ///
    /// Means the buffer handed to the GLCM was never quantised, or was
    /// quantised against a different level count. Caught up front because the
    /// alternative is an out-of-bounds index inside the counting loop.
    #[cfg(feature = "spatial-image")]
    #[error("Spatial image: quantised value {value} is out of range for {levels} GLCM levels")]
    GlcmValueOutOfRange {
        /// The offending value.
        value: u8,
        /// Number of levels the matrix was sized for.
        levels: u8,
    },
}
