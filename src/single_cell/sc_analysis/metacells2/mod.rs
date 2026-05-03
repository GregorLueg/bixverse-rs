//! Port of the MetaCells2 workflow over into `biverse-rs`. The original
//! implementations were in R/C++ for the original implementation by Baran,
//! et al., Genome Biol. 2019; this was expanded in the MetaCells2 framework
//! into Python and C++, please refer to Ben‑Kiki, et al., Genome Biol. 2022.
//! Due to complexity of the algorithm, this lives in its own module.

pub mod candidates;
pub mod deviants;
pub mod dissolve;
pub mod downsample;
pub mod knn;
pub mod params;
pub mod partition;
pub mod pile;
pub mod seeds;
pub mod select;
pub mod similarity;

// export the stuff
pub use downsample::downsample_pile;
pub use knn::build_knn_graph;
pub use params::{
    DeviantsParams, DissolveParams, MC2KnnParams, MetacellsParams, PartitionParams, SelectParams,
    SimilarityMethod, SimilarityParams,
};
pub use pile::Pile;
pub use seeds::{choose_seeds, seeds_count_for};
pub use select::select_features;
pub use similarity::compute_similarity;
