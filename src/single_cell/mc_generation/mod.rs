//! Module contains methods for the generation of metacells, namely for now
//!
//! * A bootstrapped MetaCell approach based on Morabito, et al., Cell Rep.
//!   Methods, 2023
//! * The SEACells approach from Persad, et al., Nat. Biotechnol., 2023.
//! * The SuperCell approach from Bilous, et al., BMC Bioinform., 2022.
//! * The MetaCell2 approach from Baran, et al., Genome Biol. 2019; this was
//!   expanded in the MetaCells2 framework into Python and C++, please refer to
//!   Ben‑Kiki, et al., Genome Biol. 2022.

pub mod hdwgcna_meta_cells;
pub mod metacells2;
pub mod seacells;
pub mod super_cells;
