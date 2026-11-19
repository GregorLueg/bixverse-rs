//! Re-exports for `use bixverse_atac::prelude::*`: the single cell prelude
//! plus this crate's errors and entry points.

pub use bixverse_sc::prelude::*;

pub use crate::errors::*;
pub use crate::integration::rlsi::{
    BatchLsi, RlsiParams, rlsi_anchors, rlsi_integration, rlsi_on_store,
};
pub use crate::io::annotation::{
    GeneActivityParams, GeneRecord, gene_activity_regions, read_gtf_genes, tss_positions,
};
pub use crate::io::fragments::{
    BarcodeIndex, Coverage, Fragment, barcode_fragment_counts, fold_fragments,
};
#[cfg(feature = "h5")]
pub use crate::io::tenx_peaks::{TenxPeakIngest, tenx_peaks_to_store};
pub use crate::peaks::calling::{KeepDup, NarrowPeak, PeakCallParams, TagSource, call_peaks};
pub use crate::peaks::regions::{RegionIndex, genome_tiles, read_bed, reduce_regions};
pub use crate::processing::amulet::{AmuletParams, AmuletResult, amulet};
pub use crate::processing::counting::{
    CountMode, FeatureCounts, count_fragments, write_counts_store,
};
pub use crate::processing::features::{
    FeatureTotals, TopFeatureCutoff, feature_total_stats, select_top_features,
};
pub use crate::processing::lsi::{
    LsiParams, LsiResult, TF_IDF_SCALE, lsi_on_matrix, lsi_on_store, project_lsi, read_tf_idf,
    set_tf_idf_on_store,
};
pub use crate::processing::profiles::{FragmentProfiles, fragment_profiles};
pub use crate::processing::qc::{FragmentQc, FragmentQcParams, fragment_qc};
