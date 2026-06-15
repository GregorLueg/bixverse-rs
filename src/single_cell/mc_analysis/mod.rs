//! Meta cells analysis methods. These are methods that leverage the aggregated
//! counts from the meta-cells. Focus on 'bag-of-genes' methods, such as
//! SCENIC, etc.

pub mod aucell;
pub mod metacell_density;
pub mod metrics;
pub mod nmf_mc;
pub mod scenic_metacells;
