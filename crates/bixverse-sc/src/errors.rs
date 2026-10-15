//! Errors raised by `bixverse-sc`.

use thiserror::Error;

/// Error variants raised by `bixverse-sc`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum ScErrors {
    // -- external crates --
    /// Propagates an error from `ann-search-rs`.
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),

    /// Propagates an error from `edge-rs`.
    #[error("Error from the edge-rs crate: {0}")]
    EdgeRsError(#[from] edge_rs::errors::EdgeErrors),

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

    // -- bonsai-rs --
    /// Propagate errors from the bonsai-rs crate
    #[cfg(feature = "bonsai")]
    #[error("Error from the bonsai-rs crate: {0}")]
    BonsaiRsError(#[from] bonsai_rs::errors::BonsaiErrors),

    /// Propagate errors from the sanity-sc-rs crate, as re-exported by
    /// bonsai-rs
    #[cfg(feature = "bonsai")]
    #[error("Error from the sanity-sc-rs crate: {0}")]
    SanityRsError(#[from] bonsai_rs::sanity_sc_rs::errors::SanityErrors),

    /// Most of the supplied kNN indices point outside the cell set
    ///
    /// The usual cause is a kNN computed against a reference index rather than
    /// the query set, which drops every edge and returns an all-zero
    /// connectivity matrix looking like a genuine result.
    #[error(
        "PAGA: {out_of_range} of {total} kNN entries point outside the {n_cells} cells; the graph was probably built against a different index"
    )]
    PagaKnnIndicesOutOfRange {
        /// Entries pointing past the last cell
        out_of_range: usize,
        /// Entries supplied in total
        total: usize,
        /// Cells in the input
        n_cells: usize,
    },

    /// Two stores that must share a gene axis disagree on how many genes
    /// they hold.
    ///
    /// Raised where a method takes a gene-major and a cell-major reader over
    /// what is meant to be the same experiment. Nothing ties the two files
    /// together, so the check has to be explicit: indices derived from one
    /// store are used against buffers sized from the other.
    #[error(
        "The gene store holds {gene_store} genes but the cell store holds {cell_store}; they must describe the same experiment"
    )]
    GeneAxisMismatch {
        /// Total genes reported by the gene-major store
        gene_store: usize,
        /// Total genes reported by the cell-major store
        cell_store: usize,
    },
    // -- Batch --
    /// Need at least two batches for this method
    #[error("You need at least two batches. Provided {n_batches} batches.")]
    NeedAtLeastTwoBatches {
        /// Number of provided batches
        n_batches: usize,
    },

    // -- Integration metrics --
    /// Need at least two distinct labels for this metric
    #[error("You need at least two distinct labels. Provided {n_labels} labels.")]
    NeedAtLeastTwoLabels {
        /// Number of distinct labels found
        n_labels: usize,
    },

    /// The label vector does not line up with the number of cells
    #[error("Expected {n_cells} labels, one per cell; got {n_labels}.")]
    MetricLabelLengthMismatch {
        /// Number of cells in the embedding or kNN graph
        n_cells: usize,
        /// Length of the provided label vector
        n_labels: usize,
    },

    /// A batch holds too few cells for the requested per-batch neighbour count
    #[error("BBKNN: batch {batch} has {n_cells} cells, needs at least {required}.")]
    BbknnBatchTooSmall {
        /// Batch label
        batch: usize,
        /// Cells in the batch
        n_cells: usize,
        /// Minimum required, i.e. `neighbours_within_batch + 1`
        required: usize,
    },

    /// A batch has too few cells for the requested anchor search
    #[error("Batch {batch} has {n_cells} cells, needs at least {required} for anchor finding.")]
    TooFewCellsForAnchor {
        /// Batch index
        batch: usize,
        /// Cells in the batch
        n_cells: usize,
        /// Minimum required
        required: usize,
    },
    // -- Harmony --
    /// Sigma length is not equal to the number of clusters
    #[error("Harmony: sigma length must match number of clusters")]
    HarmonySigmaLengthUnequalCluster,

    // -- PCA --
    /// Error if user wants CLR-normalised PCA, but did not provide the offsets
    #[error(
        "You did not provide the offsets needed for using the CLR-normalised PCA in single cell."
    )]
    OffsetsNotProvidedForClrPCA,

    /// Error if user wants CLR-normalised PCA, but did not provide the offsets
    #[error(
        "The provided offsets have length {len_offset}; n_cells is {n_cells}. Length mismatch."
    )]
    OffsetsLengthDoesNotMatchNCells {
        /// Length of the offsets
        len_offset: usize,
        /// Number of cells provided
        n_cells: usize,
    },

    // -- HVG --
    /// A requested cell index does not exist in the store.
    #[error("HVG: cell index {index} is outside the store, which holds {total_cells} cells.")]
    HvgCellIndexOutOfRange {
        /// The offending cell index
        index: usize,
        /// Number of cells the store holds
        total_cells: usize,
    },

    /// The same cell was selected twice, which would inflate the denominator.
    #[error("HVG: cell index {index} appears more than once in the selection.")]
    HvgDuplicateCellIndex {
        /// The duplicated cell index
        index: usize,
    },

    /// Batch labels and cell indices disagree in length.
    #[error("HVG: {n_labels} batch labels were given for {n_cells} selected cells.")]
    HvgBatchLabelLengthMismatch {
        /// Number of batch labels provided
        n_labels: usize,
        /// Number of selected cells
        n_cells: usize,
    },

    /// A batch in `0..n_batches` ended up with no cells.
    #[error("HVG: batch {batch} has no cells; labels must densely cover 0..{n_batches}.")]
    HvgEmptyBatch {
        /// The empty batch
        batch: usize,
        /// Number of batches implied by the labels
        n_batches: usize,
    },

    /// No cells were selected at all.
    #[error("HVG: the cell selection is empty.")]
    HvgEmptySelection,

    /// Loess span outside the range [`crate::core::base::loess::LoessRegression`]
    /// accepts.
    #[error("HVG: the loess span is {span}, but it must be within (0, 1].")]
    HvgInvalidLoessSpan {
        /// The provided span
        span: f32,
    },

    /// Bin count of zero, which would divide by zero during binning.
    #[error("HVG: n_bins must be at least 1.")]
    HvgInvalidBinCount,

    /// Too few genes pass the mean filter to fit the scran variance trend.
    #[error(
        "HVG: only {n_kept} genes have a mean log-expression of at least {minimum_mean}; the variance trend needs two."
    )]
    HvgTooFewTrendPoints {
        /// Genes left after the mean filter
        n_kept: usize,
        /// The mean filter threshold
        minimum_mean: f64,
    },

    // -- Hotspot --
    /// Invalid model chosen for Hotspot
    #[error("Invalid model type: {0}")]
    HotSpotWrongModel(String),

    // -- Metacells2 --
    /// User needs to have generated select_features prior to the
    /// compute_similarity part.
    #[error("select_features must be called before compute_similarity!")]
    SelectFeaturesBeforeSimilariy,

    // -- Seacells --
    /// The SEACells kernel matrix has not been yet constructed.
    #[error("SEACells: You must construct the kernel matrix first.")]
    SEACellsKernelMatrixMissing,

    /// The SEACells archetypes have not yet been identified.
    #[error("SEACells: You must first initialise the Archetypes.")]
    SEACellsArchetypesMissing,

    /// The SEACells archetypes have not yet been identified.
    #[error("SEACells: The model has not been fitted yet. Please run .fit()")]
    SEACellsModelNotFitted,

    /// The adaptive diffusion kernel came out with non-finite weights.
    ///
    /// Raised by
    /// [bixverse_sc::mc_generation::seacells::compute_diffusion_kernel].
    /// The usual cause is a caller-supplied kNN graph carrying `NaN` distances,
    /// since zero bandwidths are handled by the smallest-positive fallback.
    #[error(
        "The adaptive diffusion kernel over {n_cells} cells holds non-finite weights; check the supplied kNN distances"
    )]
    DiffusionKernelNotFinite {
        /// Cells in the kNN graph
        n_cells: usize,
    },

    // -- MELD --
    /// If the user provides a label which is out-of-range of the expected
    /// number of groups
    #[error("MELD: label {label} out of range for n_groups={n_groups}.")]
    MELDLabelOutOfRange {
        /// The label which is out of range
        label: usize,
        /// The number of expected groups
        n_groups: usize,
    },

    /// If the user asks for less than two Chebyshev coefficients
    #[error("MELD: Need at least 2 Chebyshev coefficients")]
    MELDChebyshevCoefTooLow,

    /// Error if labels and number of cells are not equal
    #[error("MELD: The labels are not equal to the number of cells")]
    MELDLabelUnequalsSamples,

    /// Error of n_groups < 2
    #[error("MELD: labels needs two groups minimum")]
    MELDOnlyOneGroup,

    // -- DGE --
    /// Fewer than two cell groups, so there is nothing to compare.
    #[error("DGE: at least two cell groups are required, got {n_groups}")]
    DgeTooFewGroups {
        /// Number of groups supplied
        n_groups: usize,
    },

    /// More cell groups than the `u16` group labels can hold.
    #[error("DGE: {n_groups} cell groups exceed the maximum of {max}")]
    DgeTooManyGroups {
        /// Number of groups supplied
        n_groups: usize,
        /// Largest supported number of groups
        max: usize,
    },

    /// One of the cell groups holds no cells.
    #[error("DGE: cell group {group} is empty")]
    DgeEmptyGroup {
        /// Index of the offending group
        group: usize,
    },

    /// A cell index beyond the end of the store.
    #[error("DGE: cell {cell} is out of range for a store of {n_cells} cells")]
    DgeCellOutOfRange {
        /// The offending cell index
        cell: usize,
        /// Total cells in the store
        n_cells: usize,
    },

    /// A cell appears in two groups, or twice in one.
    ///
    /// Silently biases every AUROC toward 0.5 rather than failing, so it is
    /// worth rejecting outright.
    #[error("DGE: cell {cell} appears in cell group {first} and again in group {second}")]
    DgeOverlappingGroups {
        /// The cell listed more than once
        cell: usize,
        /// Group the cell was first seen in
        first: usize,
        /// Group the cell was seen in again
        second: usize,
    },

    /// A reference arm index beyond the number of groups.
    #[error("DGE: reference {reference} is out of range for {n_groups} cell groups")]
    DgeReferenceOutOfRange {
        /// The offending reference index
        reference: usize,
        /// Number of groups supplied
        n_groups: usize,
    },

    // -- NEBULA --
    /// Every gene failed NEBULA's own expression filter.
    ///
    /// Raised only once the whole sweep is done. A single batch losing all of
    /// its genes is normal and is skipped silently.
    #[error(
        "NEBULA: none of the {n_genes} genes passed the expression filter; lower `cpc` or `mincp`"
    )]
    NebulaNoGenesKept {
        /// How many genes were offered
        n_genes: usize,
    },

    // -- Palantir --
    /// The user-supplied early cell index sits outside the cell range.
    #[error("Palantir: early cell index {early_cell} is out of range for {n_cells} cells")]
    PalantirEarlyCellOutOfRange {
        /// The offending early cell index
        early_cell: usize,
        /// Number of cells in the input
        n_cells: usize,
    },

    /// A user-supplied terminal state index sits outside the cell range.
    #[error("Palantir: terminal state index {state} is out of range for {n_cells} cells")]
    PalantirTerminalStateOutOfRange {
        /// The offending terminal state index
        state: usize,
        /// Number of cells in the input
        n_cells: usize,
    },

    /// A terminal state cell was never sampled as a waypoint, so it has no
    /// position in the waypoint Markov chain.
    #[error(
        "Palantir: terminal state cell {state} is not among the {n_waypoints} sampled waypoints"
    )]
    PalantirTerminalStateNotAWaypoint {
        /// The offending terminal state cell index
        state: usize,
        /// Number of sampled waypoints
        n_waypoints: usize,
    },

    /// The first waypoint is not the start cell, which every consumer of the
    /// geodesic matrix assumes.
    #[error(
        "Palantir: waypoint 0 is cell {first_waypoint}, but the start cell is {start_cell}; the start cell must be the first waypoint"
    )]
    PalantirStartCellNotFirstWaypoint {
        /// The start cell the caller asked for
        start_cell: usize,
        /// The cell actually sitting in waypoint position zero
        first_waypoint: usize,
    },

    /// The pseudotime refinement cap leaves no room for a single pass.
    #[error(
        "Palantir: max_iterations is {max_iterations}, but at least {minimum} is needed to run a refinement pass"
    )]
    PalantirMaxIterationsTooSmall {
        /// The requested cap
        max_iterations: usize,
        /// Smallest cap that runs at least one pass
        minimum: usize,
    },

    /// The neighbour count is too small to derive the back-edge bandwidth of
    /// the waypoint Markov chain, which indexes neighbour rank
    /// `min(knn / 3 - 1, 30)`.
    ///
    /// Carries the neighbour count the caller asked for, not the one clamped to
    /// the waypoint set. Too few waypoints raises
    /// [BixverseErrors::PalantirTooFewWaypoints] instead, so the message always
    /// names the parameter that can actually be changed.
    #[error("Palantir: knn is {knn}, but at least {minimum} is needed for the adaptive bandwidth")]
    PalantirKnnTooSmall {
        /// The requested neighbour count
        knn: usize,
        /// Smallest neighbour count yielding a valid bandwidth rank
        minimum: usize,
    },

    /// Too few waypoints were sampled to support the back-edge bandwidth rank,
    /// however large the requested `knn` is.
    #[error(
        "Palantir: only {n_waypoints} waypoints were sampled, but at least {minimum} are needed for the adaptive bandwidth; raise num_waypoints or supply more cells"
    )]
    PalantirTooFewWaypoints {
        /// Waypoints actually sampled
        n_waypoints: usize,
        /// Smallest waypoint count yielding a valid bandwidth rank
        minimum: usize,
    },

    /// The waypoint data and the waypoint pseudotime disagree on the waypoint
    /// count.
    #[error(
        "Palantir: {n_waypoints} waypoint rows were supplied against {n_pseudotime} pseudotime values"
    )]
    PalantirWaypointDataMismatch {
        /// Rows of multiscale data restricted to the waypoints
        n_waypoints: usize,
        /// Pseudotime values supplied alongside
        n_pseudotime: usize,
    },

    /// The stationary ranks have no spread, so the outlier cutoff separates
    /// nothing.
    ///
    /// More than half the waypoints share a stationary mass, which puts the
    /// median absolute deviation at exactly zero and the cutoff exactly on the
    /// median. Roughly half of everything then clears it, the components merge,
    /// and the branching structure collapses to a single terminal state.
    #[error(
        "Palantir: the stationary ranks over {n_waypoints} waypoints have zero spread, so terminal states cannot be separated; supply them explicitly or raise knn"
    )]
    PalantirStationaryRanksDegenerate {
        /// Waypoints in the chain
        n_waypoints: usize,
    },

    /// Cells remained unreachable from the start cell after graph repair.
    #[error(
        "Palantir: {n_unreachable} cells remain unreachable from the start cell after {repairs} repair edges; increase knn"
    )]
    PalantirDisconnectedGraph {
        /// Cells still at infinite geodesic distance
        n_unreachable: usize,
        /// Repair edges added before giving up
        repairs: usize,
    },

    /// The pseudotime collapsed to a constant, or the Silverman bandwidth was
    /// zero or non-finite, so the refinement cannot be normalised.
    #[error("Palantir: degenerate pseudotime ({reason}); the manifold is probably collapsed")]
    PalantirDegeneratePseudotime {
        /// Which quantity degenerated
        reason: &'static str,
    },

    /// The stationary power iteration lost all of its mass, which can only
    /// happen when the supplied chain is not row-stochastic or carries
    /// non-finite entries.
    #[error(
        "Palantir: the stationary power iteration lost its mass at iteration {iteration} (total mass {total:e}); the transition matrix is not row-stochastic"
    )]
    PalantirStationaryMassLost {
        /// The pass on which the mass went non-positive or non-finite
        iteration: usize,
        /// The offending total mass
        total: f64,
    },

    /// An eigenvalue of the diffusion operator came back non-finite.
    ///
    /// Clamping it would fabricate a plausible-looking scale factor out of a
    /// `NaN`, since `f64::min` returns the other operand for a `NaN` input.
    #[error(
        "Palantir: the diffusion eigensolve returned a non-finite eigenvalue ({eigenvalue}); the kernel is degenerate"
    )]
    PalantirNonFiniteEigenvalue {
        /// The offending eigenvalue
        eigenvalue: f64,
    },

    /// An explicit `n_eigs` below the smallest usable count was requested.
    ///
    /// The trivial leading eigenvector is always dropped, so fewer than two
    /// eigenvectors leaves no multiscale coordinate at all. The request is
    /// refused rather than raised to the heuristic's floor: silently changing a
    /// pinned count changes every distance downstream.
    #[error(
        "Palantir: n_eigs is {n_eigs}, but at least {minimum} eigenvectors are needed for a multiscale space"
    )]
    PalantirNEigsTooSmall {
        /// The requested eigenvector count
        n_eigs: usize,
        /// Smallest count leaving a usable coordinate
        minimum: usize,
    },

    /// No terminal states survived the rank cutoff and the caller supplied none.
    #[error(
        "Palantir: no terminal states detected; supply them explicitly or increase num_waypoints"
    )]
    PalantirNoTerminalStates,

    /// The multiscale space has no boundary cells, so there is nothing to snap
    /// the start cell or a terminal-state candidate onto.
    ///
    /// Boundaries are the per-component argmin and argmax, so an empty set means
    /// the multiscale space has no columns at all.
    #[error("Palantir: the multiscale space has no boundary cells to snap onto")]
    PalantirNoBoundaryCells,

    /// Cells remained unreachable from the waypoint set when the geodesic
    /// matrix was built.
    ///
    /// Distinct from [BixverseErrors::PalantirDisconnectedGraph], which reports
    /// the repair attempts. This fires after repair, inside the pseudotime
    /// solve, where the repair count is not in scope and claiming zero of them
    /// would be a lie.
    #[error("Palantir: {n_unreachable} cells have no geodesic path to some waypoint; increase knn")]
    PalantirUnreachableFromWaypoints {
        /// Cells at infinite geodesic distance from at least one waypoint
        n_unreachable: usize,
    },

    /// The densified `(I - Q) B = R` solve did not converge.
    ///
    /// Carries the achieved `‖(I - Q) B - R‖_inf`, which is what actually
    /// distinguishes a near-singular system from ordinary rounding.
    #[error(
        "Palantir: the (I - Q) B = R solve over {n_transient} transient waypoints left a residual of {residual:e}, above tolerance"
    )]
    PalantirAbsorbingSolveFailed {
        /// Achieved `‖(I - Q) B - R‖_inf`
        residual: f64,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The absorption probabilities of one waypoint sum to more than one.
    ///
    /// Split out from [BixverseErrors::PalantirAbsorbingSolveFailed] because the
    /// solve residual says nothing here: the factorisation can be exact to
    /// machine precision and still land on this, which is what makes the
    /// offending sum the only number worth reporting.
    #[error(
        "Palantir: waypoint {waypoint} absorbs with probability {row_sum} across {n_transient} transient waypoints, which is above one"
    )]
    PalantirAbsorbingRowSum {
        /// The offending row sum
        row_sum: f64,
        /// Waypoint index the row belongs to
        waypoint: usize,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The absorption probabilities of one waypoint are not finite.
    ///
    /// Split from [BixverseErrors::PalantirAbsorbingRowSum] because the two
    /// point at different bugs: a non-finite sum means a `NaN` reached the
    /// solve, while a sum above one means the chain was not sub-stochastic.
    #[error(
        "Palantir: waypoint {waypoint} has a non-finite absorption row sum ({row_sum}) across {n_transient} transient waypoints; the solve produced non-finite entries"
    )]
    PalantirAbsorbingRowNotFinite {
        /// The offending row sum
        row_sum: f64,
        /// Waypoint index the row belongs to
        waypoint: usize,
        /// Total transient waypoints
        n_transient: usize,
    },

    /// The densified `(I - Q)` would exceed the supported size.
    #[error(
        "Palantir: {n_transient} transient waypoints exceed the dense solve limit of {limit}; lower num_waypoints"
    )]
    PalantirTooManyWaypointsForDenseSolve {
        /// Transient waypoint count requested
        n_transient: usize,
        /// The supported ceiling
        limit: usize,
    },

    // -- MAGIC --
    /// The requested imputation would not fit in a sane amount of memory.
    ///
    /// MAGIC output is unavoidably dense, so the only bound available is the
    /// caller's gene selection. This is an error rather than a warning because
    /// the crate is driven from R, where a printed warning is easily missed and
    /// the alternative failure mode is the session being OOM-killed.
    #[error(
        "MAGIC: imputing {n_genes} genes across {n_cells} cells needs {} MB of dense f32, above the {} MB budget; subset the genes or set allow_large",
        (*n_cells as f64 * *n_genes as f64 * 4.0 / 1.048576e6).round(),
        (*max_elements as f64 * 4.0 / 1.048576e6).round()
    )]
    MagicOutputTooLarge {
        /// Cells in the selection
        n_cells: usize,
        /// Genes in the selection
        n_genes: usize,
        /// The element ceiling that was exceeded
        max_elements: usize,
    },

    /// The kNN inputs disagree with the requested cell selection.
    #[error(
        "MAGIC: the kNN graph has {n_knn} rows but {n_selected} cells were selected; the two must be aligned"
    )]
    MagicKnnSelectionMismatch {
        /// Rows in the supplied kNN graph
        n_knn: usize,
        /// Cells in the selection
        n_selected: usize,
    },

    /// A cell index in the selection sits outside the store.
    #[error("MAGIC: cell index {cell} is out of range for a store of {total_cells} cells")]
    MagicCellOutOfRange {
        /// The offending cell index
        cell: usize,
        /// Cells in the store
        total_cells: usize,
    },

    /// A cell index appears twice in the selection.
    #[error("MAGIC: cell index {cell} appears more than once in the selection")]
    MagicDuplicateCell {
        /// The duplicated cell index
        cell: usize,
    },

    /// The operator was built against a different store than the one being read.
    ///
    /// The operator's cell lookup is indexed by global cell id, so reading a
    /// store with more cells than it was sized for would index past its end.
    #[error(
        "MAGIC: the operator was built for a store of {operator_cells} cells but the reader holds {store_cells}; the `total_cells` passed to MagicOperator::from_knn must match the store"
    )]
    MagicStoreSizeMismatch {
        /// Cells the operator was built for
        operator_cells: usize,
        /// Cells the reader actually holds
        store_cells: usize,
    },

    // -- Gene trends --
    /// The expression matrix does not have one row per cell.
    #[error(
        "Gene trends: the expression matrix has {n_cells} rows but {n_pseudotime} pseudotime values were supplied"
    )]
    GeneTrendsShapeMismatch {
        /// Rows in the expression matrix
        n_cells: usize,
        /// Length of the pseudotime vector
        n_pseudotime: usize,
    },

    /// A branch selected no cells at all.
    #[error("Gene trends: branch {branch} selected no cells")]
    GeneTrendsBranchEmpty {
        /// Index of the offending branch
        branch: usize,
    },

    /// A branch has too few cells to fit anything meaningful.
    ///
    /// The reference has no such guard and will happily fit a 500-point grid to
    /// a single cell. The resulting curve is the prior, not the data.
    #[error(
        "Gene trends: branch {branch} has only {n_cells} cells, which is below the minimum of {minimum}"
    )]
    GeneTrendsBranchTooFewCells {
        /// Index of the offending branch
        branch: usize,
        /// Cells the branch selected
        n_cells: usize,
        /// Minimum the fit requires
        minimum: usize,
    },

    /// Every cell in a branch shares the same pseudotime, so there is no grid
    /// to fit over.
    #[error("Gene trends: branch {branch} spans zero pseudotime, so its grid would be degenerate")]
    GeneTrendsDegeneratePseudotime {
        /// Index of the offending branch
        branch: usize,
    },

    /// Fate-probability weighting was asked for without the probabilities.
    #[error(
        "Gene trends: BranchWeighting::FateProbability needs the fate probabilities, but none were supplied"
    )]
    GeneTrendsMissingFateProbabilities,

    /// The fate-probability matrix does not have one column per branch.
    ///
    /// Branch `i` takes column `i` positionally, so a mismatch would silently
    /// weight every cell by the wrong fate rather than failing.
    #[error(
        "Gene trends: the fate probabilities have {n_columns} columns but {n_branches} branches were supplied; branches map to columns positionally, so the two must agree"
    )]
    GeneTrendsFateColumnMismatch {
        /// Columns in the fate probability matrix
        n_columns: usize,
        /// Branches supplied
        n_branches: usize,
    },

    /// The fate-probability matrix does not have one row per cell.
    #[error(
        "Gene trends: the fate probabilities have {n_rows} rows but {n_cells} cells were supplied"
    )]
    GeneTrendsFateShapeMismatch {
        /// Rows in the fate probability matrix
        n_rows: usize,
        /// Cells supplied
        n_cells: usize,
    },

    /// A branch cell index sits outside the cell range.
    #[error(
        "Gene trends: branch {branch} references cell {cell}, which is out of range for {n_cells} cells"
    )]
    GeneTrendsCellOutOfRange {
        /// Index of the offending branch
        branch: usize,
        /// The offending cell index
        cell: usize,
        /// Cells in the input
        n_cells: usize,
    },

    /// Every cell in a branch carries zero fate-probability weight.
    ///
    /// The Gaussian process treats a zero-weight observation as absent, so this
    /// would otherwise return the zero-mean prior and be indistinguishable from
    /// a genuinely flat gene.
    #[error(
        "Gene trends: every cell selected for branch {branch} has zero fate probability, so the fit would carry no data"
    )]
    GeneTrendsZeroWeightBranch {
        /// Index of the offending branch
        branch: usize,
    },

    /// The requested grid is too coarse to fit anything over.
    #[error("Gene trends: a resolution of {resolution} is below the minimum of {minimum}")]
    GeneTrendsResolutionTooLow {
        /// Requested grid points
        resolution: usize,
        /// Minimum the fit requires
        minimum: usize,
    },

    // -- DIALOGUE --
    /// Fewer than two cell types were supplied.
    ///
    /// There is nothing to correlate across with one.
    #[error("DIALOGUE: needs at least two cell types; got {n_cell_types}.")]
    DialogueTooFewCellTypes {
        /// Cell types supplied
        n_cell_types: usize,
    },

    /// Too few samples are represented in every cell type.
    ///
    /// The decomposition runs on the samples the cell types share, and that
    /// intersection is what has come up short.
    #[error(
        "DIALOGUE: only {found} samples are present in every cell type; at least {minimum} are needed."
    )]
    DialogueTooFewSharedSamples {
        /// Samples in the intersection
        found: usize,
        /// Minimum the decomposition requires
        minimum: usize,
    },

    /// Too few features in one cell type survived the ANOVA filter.
    ///
    /// A feature is kept when it varies across samples. If almost none do, the
    /// cell type carries no sample-level signal to decompose, and the usual
    /// fixes are to drop it from the run or to set the spatial flag.
    #[error(
        "DIALOGUE: only {found} of {total} features in cell type {cell_type} passed the ANOVA filter; at least {minimum} are needed."
    )]
    DialogueTooFewFeatures {
        /// Index of the offending cell type
        cell_type: usize,
        /// Features that passed
        found: usize,
        /// Features tested
        total: usize,
        /// Minimum the decomposition requires
        minimum: usize,
    },

    /// Too few samples carry enough cells for the ANOVA filter to run.
    #[error(
        "DIALOGUE: cell type {cell_type} has {found} samples with at least {min_cells} cells; the ANOVA filter needs two. Lower abn_c or set the spatial flag."
    )]
    DialogueTooFewAbundantSamples {
        /// Index of the offending cell type
        cell_type: usize,
        /// Samples meeting the threshold
        found: usize,
        /// Cells a sample must contribute
        min_cells: usize,
    },

    // -- CellPhoneDB --
    /// A partner subunit points outside the genes in the store.
    #[error("CellPhoneDB: gene index {index} is outside the {n_genes} genes in the store.")]
    CpdbGeneIndexOutOfRange {
        /// The offending index
        index: usize,
        /// Number of genes in the store
        n_genes: usize,
    },

    /// A cluster lists a cell outside the store.
    #[error("CellPhoneDB: cell index {index} is outside the {n_cells} cells in the store.")]
    CpdbCellIndexOutOfRange {
        /// The offending index
        index: usize,
        /// Number of cells in the store
        n_cells: usize,
    },

    /// An interaction has a partner without any subunit genes.
    #[error("CellPhoneDB: interaction {interaction} has a partner without genes.")]
    CpdbEmptyPartner {
        /// Index of the offending interaction
        interaction: usize,
    },

    /// A cell was assigned to more than one cluster.
    ///
    /// The permutation shuffles one label per cell, so clusters must be
    /// disjoint.
    #[error("CellPhoneDB: cell {cell} appears in more than one cluster.")]
    CpdbClusterOverlap {
        /// The offending cell index
        cell: usize,
    },

    // -- sctype --
    /// Error when number of cluster assignment != the number of cells
    #[error(
        "SCType: The number of cells ({n_cells}) and cluster assignments length ({n_cluster_assignment}) is not the same."
    )]
    ScTypeClusterAssignmentNotEqualNCells {
        /// Number of cells
        n_cells: usize,
        /// Number of cluster assignments
        n_cluster_assignment: usize,
    },

    /// Error when the smoothing graph node count != the number of cells
    #[error("SCType: The graph has {n_nodes} nodes, but there are {n_cells} cells.")]
    ScTypeGraphNodesNotEqualNCells {
        /// Number of cells
        n_cells: usize,
        /// Number of nodes in the smoothing graph
        n_nodes: usize,
    },

    // -- scTransform --
    /// A step-1 gene index pointed outside the modelled gene set.
    #[error("scTransform: step-1 gene index {index} is outside the {n_genes} modelled genes.")]
    SctGeneIndexOutOfRange {
        /// The offending index
        index: usize,
        /// Number of modelled genes
        n_genes: usize,
    },

    /// Too few step-1 genes survived outlier and Poisson exclusion to smooth.
    ///
    /// Two is the bare minimum a bandwidth can be estimated from. In practice
    /// this means the subsample was dominated by Poisson genes, which is what a
    /// very shallow dataset looks like.
    #[error(
        "scTransform: only {kept} of {total} step-1 genes survived exclusion, too few to regularise."
    )]
    SctTooFewGenesToRegularise {
        /// Genes left after exclusion
        kept: usize,
        /// Genes the step-1 fit started from
        total: usize,
    },

    /// The kernel smoothing produced no value for a gene.
    ///
    /// Means no retained step-1 gene fell inside the kernel support at that
    /// gene's abundance, which a bandwidth this wide should make impossible.
    #[error("scTransform: regularisation left gene {gene} without a fitted parameter.")]
    SctSmoothingLeftAGeneUnfitted {
        /// Position of the gene in the modelled set
        gene: usize,
    },

    /// No gene passed the `min_cells` detection filter.
    #[error("scTransform: no gene of {n_genes} is detected in at least {min_cells} cells.")]
    SctNoGenesPassFilter {
        /// The filter that rejected everything
        min_cells: usize,
        /// Genes the filter was applied to
        n_genes: usize,
    },

    /// A gene handed to the residual PCA is not covered by the model.
    #[error("scTransform: gene {gene} is not covered by the fitted model.")]
    SctGeneNotModelled {
        /// Store index of the gene
        gene: usize,
    },

    /// Pearson residuals and the shifted CLR transformation were both asked
    /// for.
    ///
    /// CLR rewrites the stored log-normalised layer, which the residual path
    /// never reads, so combining them would quietly do nothing.
    #[error("PCA on Pearson residuals cannot be combined with the CLR transformation.")]
    PcaResidualsWithClr,

    /// Covariate columns handed in disagree in length.
    #[error("scTransform covariate '{name}': expected {expected} values, got {found}.")]
    SctCovariateLengthMismatch {
        /// Name of the offending covariate
        name: String,
        /// Length the other columns had
        expected: usize,
        /// Length this one had
        found: usize,
    },

    /// The covariates supplied do not match what the model was fitted with.
    #[error("scTransform model was fitted with {model} covariate(s) but {supplied} were supplied.")]
    SctCovariateCountMismatch {
        /// Covariates the model carries
        model: usize,
        /// Covariates the caller supplied
        supplied: usize,
    },

    // -- residual normalisation --
    /// Group labels disagree in length with the selected cells.
    #[error("Residual grouping: {n_labels} group label(s) for {n_cells} cell(s).")]
    ResidualGroupLabelLengthMismatch {
        /// Labels supplied
        n_labels: usize,
        /// Cells selected
        n_cells: usize,
    },

    /// A group id has no cells in it.
    ///
    /// Labels must densely cover `0..n_groups`. A gap means the caller coded a
    /// factor without dropping its unused levels, and a model would be fitted
    /// against nothing.
    #[error("Residual grouping: group {group} of {n_groups} has no cells.")]
    ResidualEmptyGroup {
        /// The empty group id
        group: usize,
        /// Groups the labels imply
        n_groups: usize,
    },

    /// No gene is modelled in every group.
    ///
    /// The shared feature axis is the intersection of the per-group modelled
    /// sets, so an empty intersection means the groups have no gene in common
    /// that passes `min_cells` everywhere. Usually one very shallow sample.
    #[error("Residual grouping: no gene is modelled in all {n_groups} groups.")]
    ResidualEmptyGeneIntersection {
        /// Groups that were intersected
        n_groups: usize,
    },

    /// The count block handed to the analytic Pearson model sums to zero.
    ///
    /// `mu_cg = n_c * p_g` divides by the grand total, so a block of pure
    /// zeros has no model at all rather than a degenerate one.
    #[error("Analytic Pearson residuals: the selected counts sum to zero.")]
    AnalyticPearsonZeroTotal,

    /// A non-positive overdispersion was supplied.
    #[error("Analytic Pearson residuals: theta must be positive, got {theta}.")]
    AnalyticPearsonInvalidTheta {
        /// The offending value
        theta: f64,
    },

    /// An unusable residual clipping range was supplied.
    ///
    /// `f64::clamp` panics when the bounds are inverted or either is NaN, so
    /// this has to be caught before a rayon worker reaches it.
    #[error("Residual clipping range [{lo}, {hi}] must be finite with lo <= hi.")]
    ResidualInvalidClipRange {
        /// Lower bound supplied
        lo: f64,
        /// Upper bound supplied
        hi: f64,
    },

    /// The selected cells contain a repeated index.
    ///
    /// The readers key their selection by an `IndexSet`, which deduplicates,
    /// while the residual row is sized by the raw selection. A duplicate would
    /// therefore shift every later cell into the wrong slot rather than
    /// erroring.
    #[error("Residual selection: {n_cells} cell(s) selected but only {n_unique} are distinct.")]
    ResidualDuplicateCells {
        /// Distinct cells in the selection
        n_unique: usize,
        /// Cells supplied
        n_cells: usize,
    },

    /// A gene set that has to be ascending is not.
    ///
    /// Gene lookups are binary searches, so an unsorted axis silently returns
    /// the wrong position rather than failing.
    #[error("Residual model: '{name}' must be strictly ascending.")]
    ResidualGenesNotAscending {
        /// What was out of order
        name: &'static str,
    },

    /// The group labels imply a different number of groups than there are
    /// models.
    #[error("Residual grouping: labels imply {implied} group(s) but {models} model(s) were given.")]
    ResidualGroupModelCountMismatch {
        /// Groups the labels imply
        implied: usize,
        /// Models supplied
        models: usize,
    },

    /// A group label is outside the range the models cover.
    #[error("Residual grouping: label {label} is outside the {n_groups} group(s) available.")]
    ResidualGroupLabelOutOfRange {
        /// The offending label
        label: usize,
        /// Groups available
        n_groups: usize,
    },

    /// A residual variance came out non-finite.
    ///
    /// Always an upstream problem, typically a model deserialised with a zero
    /// or missing scalar. Reported rather than floored, because a floored NaN
    /// silently ranks the gene last in feature selection.
    #[error("Residual variance for gene {gene} is not finite ({value}).")]
    ResidualNonFiniteVariance {
        /// Store index of the gene
        gene: usize,
        /// The offending value
        value: f64,
    },

    /// The residual row's counts and indices disagree in length.
    #[error("Residual row: {n_counts} count(s) for {n_indices} cell index/indices.")]
    ResidualRowLengthMismatch {
        /// Counts supplied
        n_counts: usize,
        /// Indices supplied
        n_indices: usize,
    },

    /// A residual row's cell index points outside the selected cells.
    #[error("Residual row: cell index {index} is outside the {n_cells} selected cell(s).")]
    ResidualCellIndexOutOfRange {
        /// The offending index
        index: usize,
        /// Cells selected
        n_cells: usize,
    },

    /// Feature selection was asked for no genes, or handed no groups.
    #[error("Residual feature selection: {what}.")]
    ResidualEmptySelectionRequest {
        /// What was empty
        what: &'static str,
    },

    /// The overdispersion search failed at every point it evaluated.
    ///
    /// Means the adjusted profile likelihood could not be computed anywhere in
    /// the bracket, typically a design that is rank-deficient for this gene's
    /// weights. The search would otherwise have minimised a constant and
    /// returned a fit built on a meaningless dispersion.
    #[error("scTransform: the dispersion search failed at every point: {reason}")]
    SctDispersionSearchFailed {
        /// The first underlying error, rendered
        reason: String,
    },

    /// One group's fit failed.
    ///
    /// Wraps the underlying error so that a failure in group 7 of 8 is not
    /// indistinguishable from one in group 0. The usual cause is a single
    /// shallow sample where no gene clears `min_cells`.
    #[error("Residual grouping: group {group} failed to fit: {reason}")]
    ResidualGroupFitFailed {
        /// The group that failed
        group: usize,
        /// The underlying error, rendered
        reason: String,
    },

    /// The covariates supplied are not the ones the model was fitted with.
    ///
    /// Only the count is cheap to check, and the count matching is exactly the
    /// case where a reordered data frame goes unnoticed, so the names are
    /// compared too.
    #[error(
        "scTransform covariate {position}: model was fitted with '{model}' but '{supplied}' was supplied."
    )]
    SctCovariateNameMismatch {
        /// Position in the design, excluding the intercept
        position: usize,
        /// Name the model carries
        model: String,
        /// Name the caller supplied
        supplied: String,
    },

    /// A cell index handed to scTransform is outside the selection.
    #[error("scTransform: cell index {index} is outside the {n_cells} selected cell(s).")]
    SctCellIndexOutOfRange {
        /// The offending index
        index: usize,
        /// Cells selected
        n_cells: usize,
    },

    /// A fitted model carries no design columns at all.
    #[error("scTransform model has n_coef = 0; at least the intercept is required.")]
    SctModelWithoutCoefficients,

    /// Variance normalisation was asked for on the residual PCA path.
    ///
    /// Residuals carry the biological signal as variance, which is the point of
    /// the transformation. Dividing it back out per gene flattens exactly the
    /// ranking the transformation produces, so this is refused rather than
    /// silently honoured.
    #[error(
        "PCA on Pearson residuals cannot normalise the variance; residuals already carry it as signal."
    )]
    PcaResidualsWithVarianceNormalisation,

    // -- cellsweep --
    /// The caller asked for a supplied empty droplet mask but did not give one.
    #[error("CellSweep: The empty droplet mask was not supplied but was requested.")]
    CellSweepEmptyMaskMissing,

    /// The expected cell count cannot be turned into a library size cutoff.
    #[error("CellSweep: Cannot take {expected} expected cells from {barcodes} barcodes.")]
    CellSweepBadExpectedCells {
        /// Number of expected cells requested
        expected: usize,
        /// Number of barcodes available
        barcodes: usize,
    },

    /// The knee detector could not find a cutoff.
    #[error("CellSweep: Cannot locate a knee from {barcodes} barcodes above the minimum count.")]
    CellSweepKneeNotFound {
        /// Number of barcodes that survived the minimum count filter
        barcodes: usize,
    },

    /// A sample has too few empty droplets to estimate an ambient profile.
    #[error(
        "CellSweep: Sample '{sample_id}' has {found} empty droplets, at least {required} are needed to estimate the ambient profile. Ingest with permissive QC cutoffs so the empty droplets survive, or set freeze_ambient_profile = false."
    )]
    CellSweepTooFewEmptyDroplets {
        /// Identifier of the offending sample
        sample_id: String,
        /// Empty droplets found
        found: usize,
        /// Empty droplets required
        required: usize,
    },

    /// A sample carries no annotated barcodes.
    #[error("CellSweep: Sample '{sample_id}' has no annotated barcodes to fit.")]
    CellSweepNoRealCells {
        /// Identifier of the offending sample
        sample_id: String,
    },

    /// A cell-type code points outside the profile matrix.
    #[error(
        "CellSweep: Sample '{sample_id}' has cell-type code {code} but only {n_celltypes} cell types."
    )]
    CellSweepCelltypeOutOfRange {
        /// Identifier of the offending sample
        sample_id: String,
        /// The offending cell-type code
        code: usize,
        /// Number of cell types declared
        n_celltypes: usize,
    },

    /// The label vector does not line up with the barcode vector.
    #[error(
        "CellSweep: Sample '{sample_id}' has {n_cells} real barcodes but {n_labels} cell-type labels."
    )]
    CellSweepLabelLengthMismatch {
        /// Identifier of the offending sample
        sample_id: String,
        /// Number of real barcodes
        n_cells: usize,
        /// Number of labels given
        n_labels: usize,
    },

    /// `freeze_empties = false` is not supported.
    #[error(
        "CellSweep: freeze_empties = false is not supported. The reference gives empty droplets a cell-type component they have no label for, which indexes past the end of the profile matrix and wraps onto the last cell type."
    )]
    CellSweepFreezeEmptiesUnsupported,

    /// The EM produced a non-finite log-likelihood.
    #[error("CellSweep: Sample '{sample_id}' diverged at iteration {iteration}.")]
    CellSweepDiverged {
        /// Identifier of the offending sample
        sample_id: String,
        /// Iteration at which the log-likelihood went non-finite
        iteration: usize,
    },

    // -- wnn --
    /// Error if the modalities do not have the same number of cells
    #[error("WNN: Both modalities need to have the same cell numbers.")]
    WNNModalitySampleMismatch,

    /// Error if k_nn > knn_range
    #[error("WNN: k_nn must be <= knn_range")]
    WNNKnnLargerThanKnnRange,

    /// Error if sigma_idx >= knn_range
    #[error("WNN: sigma_idx must be < knn_range")]
    WNNSigmaIdxOutOfRange,

    /// Error if s_nn > knn_range
    #[error("WNN: s_nn must be <= knn_range")]
    WNNSnnLargerThanKnnRange,

    /// Error if a modality's kNN row count does not match n_cells
    #[error("WNN: modality {modality} kNN row count {found} does not match n_cells {expected}")]
    WNNKnnRowCountMismatch {
        /// 0 or 1
        modality: usize,
        /// Expected count
        expected: usize,
        /// Found count
        found: usize,
    },

    /// Error if a modality's kNN distance row count does not match n_cells
    #[error(
        "WNN: modality {modality} kNN distance row count {found} does not match n_cells {expected}"
    )]
    WNNKnnDistRowCountMismatch {
        /// 0 or 1
        modality: usize,
        /// Expected count
        expected: usize,
        /// Found count
        found: usize,
    },

    /// Error if a modality has fewer than knn_range neighbours per cell
    #[error(
        "WNN: modality {modality} needs at least {expected} neighbours per cell, found {found}"
    )]
    WNNInsufficientNeighbours {
        /// 0 or 1
        modality: usize,
        /// knn_range
        expected: usize,
        /// Actual neighbour count in row 0
        found: usize,
    },
}
