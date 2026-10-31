//! The wgpu device: probing for an adapter, checking that plane operations
//! actually run, and handing a device to a GPU call.

use cubecl::Device;
use cubecl::wgpu::WgpuDevice;
use pyo3::prelude::*;
use std::panic::{self, AssertUnwindSafe};
use std::sync::OnceLock;

use crate::error::BixverseError;

/// Check plane (subgroup) operations with one plane reduction.
mod plane_check {
    use cubecl::prelude::*;
    use cubecl_utils_rs::prelude::*;

    /// Writes one plane's width, counted by a plane reduction, and the width
    /// the device reports.
    ///
    /// ### Params
    ///
    /// * `out` - Two slots: the reduced count and `PLANE_DIM`
    #[cube(launch_unchecked)]
    fn plane_width(out: &mut Tensor<u32>) {
        let n = plane_sum(1u32);
        if UNIT_POS_X == 0 {
            out[0] = n;
            out[1] = PLANE_DIM;
        }
    }

    /// Run [`plane_width`] on one plane.
    ///
    /// ### Params
    ///
    /// * `client` - The compute client
    ///
    /// ### Returns
    ///
    /// `true` if the reduction ran and agrees with the plane width.
    pub(crate) fn plane_ops_run(client: &Client) -> bool {
        let limits = GpuLimits::from_client(client);
        let Ok(out) = GpuTensor::<u32>::from_slice(&[0, 0], vec![2], client) else {
            return false;
        };
        unsafe {
            plane_width::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(limits.plane_size_max.max(1)),
                out.into_tensor_arg(),
            );
        }
        match out.read(client) {
            Ok(got) => got[0] != 0 && got[0] == got[1],
            Err(_) => false,
        }
    }
}

/// Whether a usable wgpu adapter is present
///
/// Initialises the same client every GPU function goes through, so `True`
/// means those functions will run rather than that a device merely exists.
/// Cached for the process; the client stays warm.
///
/// ### Returns
///
/// `True` when a wgpu adapter could be initialised.
#[pyfunction]
pub fn gpu_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // cubecl's adapter request ends in an `expect`, so a missing GPU
        // arrives as a panic, and catching it is the only probe there is. The
        // hook is swapped out so the expected answer on a CPU-only machine does
        // not print a backtrace.
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let probe = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = Device::from(WgpuDevice::default()).client();
        }));
        panic::set_hook(hook);
        probe.is_ok()
    })
}

/// Whether the adapter runs plane (subgroup) operations
///
/// Advertising them is not enough: the paravirtualised GPU of a macOS VM
/// reports planes but silently drops every dispatch that uses one. This runs
/// one plane reduction and checks the answer. Cached for the process.
///
/// ### Returns
///
/// `True` when an adapter is present and a plane reduction on it returns the
/// plane width.
#[pyfunction]
pub fn gpu_plane_ops() -> bool {
    static PLANE_OPS: OnceLock<bool> = OnceLock::new();
    gpu_available()
        && *PLANE_OPS.get_or_init(|| {
            plane_check::plane_ops_run(&Device::from(WgpuDevice::default()).client())
        })
}

/// Raise unless a wgpu adapter is present.
///
/// ### Returns
///
/// `Ok(())`, or a `BixverseError` naming the missing adapter.
pub(crate) fn ensure_gpu() -> PyResult<()> {
    if gpu_available() {
        Ok(())
    } else {
        Err(BixverseError::new_err(
            "no usable GPU adapter found. bixverse-gpu needs a working wgpu adapter; check \
             the GPU drivers. Probe with `bixverse_gpu.gpu_available()`.",
        ))
    }
}

/// Raise unless the adapter runs plane operations.
///
/// For the kernels that reduce within a plane (NEBULA, Sanity, Leiden): on a
/// device that drops those dispatches they would return whatever their output
/// buffers held rather than fail.
///
/// ### Returns
///
/// `Ok(())`, or a `BixverseError` pointing at the CPU twin.
pub(crate) fn ensure_plane_ops() -> PyResult<()> {
    ensure_gpu()?;
    if gpu_plane_ops() {
        Ok(())
    } else {
        Err(BixverseError::new_err(
            "this GPU adapter advertises plane (subgroup) operations but does not run them, \
             as the paravirtualised GPU of a macOS VM does. Use the CPU version instead.",
        ))
    }
}

/// Run a GPU call on the default device and release its memory pool after.
///
/// The pools are not released by the crates, and leaving a large call's
/// buffers behind starves the next one.
///
/// ### Params
///
/// * `f` - The call, handed the device
///
/// ### Returns
///
/// Whatever `f` returns.
pub(crate) fn on_device<T, E>(f: impl FnOnce(Device) -> Result<T, E>) -> Result<T, E> {
    let device: Device = WgpuDevice::default().into();
    let res = f(device.clone());
    let _ = device.client().memory_cleanup();
    res
}
