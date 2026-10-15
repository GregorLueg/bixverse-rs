//! Errors raised by `bixverse-methods`.

use thiserror::Error;

/// Error variants raised by `bixverse-methods`.
///
/// Grouped by subsystem.
#[derive(Debug, Error)]
pub enum MethodsErrors {
    // -- external crates --
    /// Propagates an error from `ann-search-rs`.
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),

    // -- lower crates --
    /// Propagates an error from `bixverse-core`.
    #[error(transparent)]
    Core(#[from] bixverse_core::errors::CoreErrors),

    // -- edge-rs --
    /// Propagate errors from the edge-rs crate
    #[cfg(feature = "dge")]
    #[error("Error from the edge-rs crate: {0}")]
    EdgeRsError(#[from] edge_rs::errors::EdgeErrors),

    /// More than one coefficient was handed to the limma route.
    ///
    /// `topTable` tabulates a single column. Testing several coefficients at
    /// once is the moderated F question, which needs `top_table_f` rather than
    /// this chain.
    #[cfg(feature = "dge")]
    #[error(
        "limma tests a single coefficient or contrast at a time; {n_coef} coefficients were given"
    )]
    LimmaMultiCoef {
        /// Number of coefficients the caller asked to test together
        n_coef: usize,
    },

    /// An input vector does not line up with the observations being modelled.
    ///
    /// Covers the design rows, the subject labels and the offsets, all of which
    /// have to be as long as the cell or sample selection.
    #[cfg(feature = "dge")]
    #[error("Differential expression: {name} has {got} entries but {expected} were expected")]
    DgeShapeMismatch {
        /// Which input is the wrong length
        name: &'static str,
        /// What the selection implies
        expected: usize,
        /// What was handed over
        got: usize,
    },

    // -- graph based errors ---
    /// Error for algorithms that expect undirected graphs
    ///
    /// For methods that expect an undirected graph, but received a directed
    /// one.
    #[error("The Graph is directed but needs to be undirected for this algorithm.")]
    GraphDirectedError,

    /// A node index in an edge list points outside the graph
    #[error("Graph: {name} index {index} is out of range for {n_nodes} nodes.")]
    GraphIndexOutOfRange {
        /// Which input carried the index
        name: &'static str,
        /// The offending index
        index: usize,
        /// Number of nodes in the graph
        n_nodes: usize,
    },

    /// A per-node or per-edge input does not line up with the graph
    #[error("Graph: {name} has length {got}, but {expected} was expected.")]
    GraphLengthMismatch {
        /// Which input is the wrong length
        name: &'static str,
        /// Length the graph implies
        expected: usize,
        /// Length that was supplied
        got: usize,
    },

    /// A node type present in the graph has no type weight
    #[error("Graph: node type '{0}' has no type weight.")]
    MissingNodeTypeWeight(String),

    /// Error for community membership/graph node number mismatch
    ///
    /// In cases where the graph and the community membership do not agree.
    #[error("The number of nodes and membership assignments in the communities do not add up.")]
    CommunityAssignmentMismatch,

    /// Error if a partition label sits outside the declared partition count
    #[error("The partition label {label} is out of range for {n_partitions} partitions.")]
    PartitionLabelOutOfRange {
        /// The offending label
        label: usize,
        /// Number of declared partitions
        n_partitions: usize,
    },

    /// Error if the partition count exceeds what the dense coarsening accepts
    #[error(
        "The number of partitions ({n_partitions}) exceeds the maximum of {max} for graph abstraction."
    )]
    TooManyPartitions {
        /// Provided number of partitions
        n_partitions: usize,
        /// The supported maximum
        max: usize,
    },

    // -- NMF --
    /// NMF Rank is too large for the NNDSVD initialisation
    #[error(
        "The requested NMF ({requested}) rank is too large for NNDSVD initialisation (available: {available})."
    )]
    NmfRankTooLarge {
        /// Requested rank
        requested: usize,
        /// Maximum available rank
        available: usize,
    },

    /// Error if NMF values are not finite
    #[error("The NMF values are not finite")]
    NmfNonFinite,

    /// Error if NMF values are negative
    #[error("Negative values were discovered for NMF. Please check the inputs.")]
    NmfNonNegativeViolated,

    /// Consensus NMF received a rank that cannot support a silhouette.
    #[error("Consensus NMF needs k >= 2, but k = {k} was requested.")]
    NmfConsensusInvalidK {
        /// Requested rank
        k: usize,
    },

    /// Consensus NMF received too few restarts to pool over.
    #[error("Consensus NMF needs at least 2 restarts, but n_runs = {n_runs} was requested.")]
    NmfConsensusTooFewRuns {
        /// Requested number of restarts
        n_runs: usize,
    },

    /// The outlier filter removed so many components that k-means cannot run.
    #[error(
        "Consensus NMF: only {n_surviving} components survived the density filter, which is fewer than k = {k}. Raise the density threshold or add restarts."
    )]
    NmfConsensusTooFewComponents {
        /// Requested rank
        k: usize,
        /// Components left after filtering
        n_surviving: usize,
    },

    /// k-means returned an empty cluster, so the consensus factor would carry a
    /// zero row and the refit would be degenerate.
    #[error(
        "Consensus NMF: cluster {cluster} of {k} is empty, so the consensus factor is degenerate."
    )]
    NmfConsensusEmptyCluster {
        /// Index of the empty cluster
        cluster: usize,
        /// Requested rank
        k: usize,
    },

    /// The k sweep was handed nothing to sweep over.
    #[error("The NMF k sweep was given an empty k range.")]
    NmfKSweepEmptyRange,

    /// A consensus step was applied to restarts run at a different rank.
    #[error(
        "Consensus NMF was asked for k = {requested}, but the restarts were run at k = {restarts}."
    )]
    NmfConsensusKMismatch {
        /// Rank requested for the consensus
        requested: usize,
        /// Rank the restarts actually hold
        restarts: usize,
    },

    /// A frozen factor handed to a refit does not line up with V.
    #[error(
        "NMF refit: the frozen factor has a shared dimension of {found}, but V expects {expected}."
    )]
    NmfDimensionMismatch {
        /// Dimension V expects
        expected: usize,
        /// Dimension the frozen factor provides
        found: usize,
    },

    // -- LDA --
    /// The requested topic count cannot be supported by the corpus.
    ///
    /// A topic needs somewhere to put its mass, so `k` above the vocabulary
    /// size leaves topics that no document can ever distinguish.
    #[error(
        "LDA: the requested topic count ({requested}) exceeds what the corpus supports (max: {max_available})."
    )]
    LdaInvalidTopicCount {
        /// Requested number of topics
        requested: usize,
        /// Largest topic count the corpus supports
        max_available: usize,
    },

    /// A Dirichlet hyperparameter is not strictly positive.
    #[error("LDA: hyperparameter '{name}' must be strictly positive, got {value}.")]
    LdaInvalidHyperparameter {
        /// Name of the offending hyperparameter
        name: String,
        /// Value that was supplied
        value: f64,
    },

    /// The document-term matrix carries no counts at all.
    #[error("LDA: the document-term matrix holds no non-zero entries.")]
    LdaEmptyMatrix,

    /// A non-finite value reached the variational parameters.
    #[error("LDA: the variational parameters became non-finite. Please check the inputs.")]
    LdaNonFinite,

    /// Coherence was asked for more top terms than the vocabulary holds.
    #[error(
        "LDA: coherence requested the top {requested} terms, but the vocabulary holds {vocab_size}."
    )]
    LdaTopNTooLarge {
        /// Requested number of top terms per topic
        requested: usize,
        /// Vocabulary size available
        vocab_size: usize,
    },

    /// Metric inputs disagree on the number of documents or topics.
    #[error("LDA metrics: expected a dimension of {expected}, but received {got}.")]
    LdaDimensionMismatch {
        /// Dimension the model implies
        expected: usize,
        /// Dimension that was supplied
        got: usize,
    },
}
