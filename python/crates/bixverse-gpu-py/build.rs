//! Fails the build when the `mlx` feature is on but mlx-c is missing.
//!
//! ann-search-rs only warns and compiles its MLX code out without `libmlxc`,
//! and bixverse-mlx follows it. Here the feature is an explicit request for
//! the MLX twins, so a silent fallback would produce a module without them.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_MLXC_AVAILABLE");
    if std::env::var_os("CARGO_FEATURE_MLX").is_some()
        && std::env::var_os("DEP_MLXC_AVAILABLE").is_none()
    {
        panic!(
            "the `mlx` feature needs mlx-c: `brew install mlx-c`, or point MLX_C_PREFIX at \
             an install prefix. Apple Silicon only."
        );
    }
}
