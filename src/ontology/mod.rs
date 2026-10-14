//! Methods that work on ontologies, think Gene Ontology or Disease/Phenotype
//! ontologies with relationships like ancestry, part of, etc.

pub mod go_elim;
#[cfg(feature = "r")]
pub mod ontology_r_wrappers;
pub mod similarity;
