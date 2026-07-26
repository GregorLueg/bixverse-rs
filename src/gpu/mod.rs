//! Contains GPU-accelerated methods via cubecl and burn.

pub mod gpu_r_wrappers;
pub mod linalg;
pub mod ml;
#[cfg(feature = "single-cell")]
pub mod sc_gpu;

use cubecl::prelude::{CubeCount, Runtime};

use crate::errors::BixverseErrors;

////////////
// Consts //
////////////

/// Smaller work group version with 32
pub const WORKGROUP_32: u32 = 32;

/// Medium work group version with 32
pub const WORKGROUP_64: u32 = 64;

/// Larger work group version with 128
pub const WORKGROUP_128: u32 = 128;

/// Even larger work group version with 256
pub const WORKGROUP_256: u32 = 256;

/// Very large work group version with 512
pub const WORKGROUP_512: u32 = 512;

/////////////
// Helpers //
/////////////

/// Build a static cube count, checked against the device's per-dimension limit.
///
/// A dispatch that busts the limit is not a soft failure: the launch is
/// rejected on the cubecl server thread, that thread dies, and every
/// subsequent call on the client returns an unrelated `CallError` from
/// somewhere else entirely. Catching it here turns that into a typed error
/// naming the kernel.
///
/// The limit comes from `R::max_cube_count()` rather than a hardcoded
/// constant, which is 65535 per dimension on wgpu but is not a portable
/// number.
///
/// ### Params
///
/// * `kernel` - Kernel name, for the error message only
/// * `x` - Requested cubes along x
/// * `y` - Requested cubes along y
/// * `z` - Requested cubes along z
///
/// ### Returns
///
/// `CubeCount::Static(x, y, z)`, or `GpuCubeCountExceeded` if any dimension is
/// over the device limit.
pub fn checked_cube_count<R: Runtime>(
    kernel: &'static str,
    x: u32,
    y: u32,
    z: u32,
) -> Result<CubeCount, BixverseErrors> {
    let limit = R::max_cube_count();
    if x > limit.0 || y > limit.1 || z > limit.2 {
        return Err(BixverseErrors::GpuCubeCountExceeded {
            kernel,
            requested: (x, y, z),
            limit,
        });
    }
    Ok(CubeCount::Static(x, y, z))
}
