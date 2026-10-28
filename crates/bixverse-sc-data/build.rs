//! Turns the linked HDF5 version into cfgs for the direct chunk reader.
//!
//! `hdf5-metno-sys` only declares `H5Dget_chunk_info_by_coord` from 1.10.5
//! and `H5Dchunk_iter` from 1.14.0, and exports the linked version to direct
//! dependents as `DEP_HDF5_VERSION_*`. Without the `h5` feature neither is
//! set and nothing references them.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(hdf5_chunk_info)");
    println!("cargo::rustc-check-cfg=cfg(hdf5_chunk_iter)");
    if std::env::var_os("DEP_HDF5_VERSION_1_10_5").is_some() {
        println!("cargo::rustc-cfg=hdf5_chunk_info");
    }
    if std::env::var_os("DEP_HDF5_VERSION_1_14_0").is_some() {
        println!("cargo::rustc-cfg=hdf5_chunk_iter");
    }
}
