//! Re-exports for `use bixverse_atac::prelude::*`: the single cell prelude
//! plus this crate's errors and entry points.

pub use bixverse_sc::prelude::*;

pub use crate::annotation::{
    GeneActivityParams, GeneRecord, gene_activity_regions, read_gtf_genes, tss_positions,
};
pub use crate::counting::{CountMode, FeatureCounts, count_fragments, write_counts_store};
pub use crate::errors::*;
pub use crate::features::{
    FeatureTotals, TopFeatureCutoff, feature_total_stats, select_top_features,
};
pub use crate::fragments::{BarcodeIndex, Fragment, fold_fragments};
pub use crate::lsi::{
    LsiParams, LsiResult, TF_IDF_SCALE, lsi_on_matrix, lsi_on_store, project_lsi, read_tf_idf,
    set_tf_idf_on_store,
};
pub use crate::qc::{FragmentQc, FragmentQcParams, fragment_qc};
pub use crate::regions::{RegionIndex, genome_tiles, read_bed, reduce_regions};
pub use crate::rlsi::{BatchLsi, RlsiParams, rlsi_anchors, rlsi_integration, rlsi_on_store};
#[cfg(feature = "h5")]
pub use crate::tenx_peaks::{TenxPeakIngest, tenx_peaks_to_store};
