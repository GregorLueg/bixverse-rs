//! Turns on `cfg(mlx_available)` for the experimental `mlx` feature.
//!
//! ann-search-rs owns the `links = "mlxc"` key and does the linking: only one
//! crate per build may declare it. Its build script exports
//! `DEP_MLXC_AVAILABLE=1` when the feature is on and `libmlxc.dylib` was found,
//! and warns otherwise, in which case the MLX code here is not compiled.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_MLXC_AVAILABLE");
    println!("cargo::rustc-check-cfg=cfg(mlx_available)");
    if std::env::var_os("CARGO_FEATURE_MLX").is_some()
        && std::env::var_os("DEP_MLXC_AVAILABLE").is_some()
    {
        println!("cargo:rustc-cfg=mlx_available");
    }
}
