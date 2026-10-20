//! Linear algebra – accelerated on the GPU. Specific algorithms that are useful
//! to run on the GPU. At the moment supports sparse SVD (randomised and block
//! Lanczos), pairwise correlation, CholeskyQR2 and the two GEMM shapes the
//! crate's own kernels cover better than a general matmul does.

pub mod cholesky_gpu;
pub mod corr;
pub mod gram;
pub mod skinny_gemm;
pub mod sparse_gpu;
pub mod sparse_lanczos_gpu;
pub mod sparse_rand_svd_gpu;
pub mod spmm;
