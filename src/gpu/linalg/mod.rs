//! Linear algebra – accelerated on the GPU. Specific algorithms that are useful
//! to run on the GPU. At the moment supports sparse SVD.

pub mod cholesky_gpu;
pub mod sparse_gpu;
pub mod sparse_rand_svd_gpu;
pub mod spmm;
