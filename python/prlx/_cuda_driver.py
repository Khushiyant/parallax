"""Thin ctypes wrapper around CUDA Driver API functions needed for module binding.

Used by _module_binder.py to write device globals into Triton JIT-compiled CUmodules.
"""

import ctypes
import sys
from typing import Tuple

# Load the CUDA driver library (libcuda.so.1)
try:
    _libcuda = ctypes.CDLL("libcuda.so.1")
except OSError:
    _libcuda = None

CUDA_SUCCESS = 0


class CUDADriverError(RuntimeError):
    """Raised when a CUDA Driver API call fails."""
    pass


def _check(result: int, func_name: str):
    if result != CUDA_SUCCESS:
        raise CUDADriverError(f"{func_name} failed with error code {result}")


def available() -> bool:
    """Return True if the CUDA driver library is loaded."""
    return _libcuda is not None


def cuFuncGetModule(cu_function: int) -> int:
    """Get the CUmodule that a CUfunction belongs to.

    Args:
        cu_function: CUfunction handle (opaque uint64).

    Returns:
        CUmodule handle (opaque uint64).
    """
    module = ctypes.c_void_p()
    result = _libcuda.cuFuncGetModule(
        ctypes.byref(module),
        ctypes.c_void_p(cu_function),
    )
    _check(result, "cuFuncGetModule")
    return module.value


def cuModuleGetGlobal(cu_module: int, name: str) -> Tuple[int, int]:
    """Get the device pointer and size of a global variable in a CUmodule.

    Args:
        cu_module: CUmodule handle.
        name: Name of the global variable (e.g. "g_prlx_buffer").

    Returns:
        (device_ptr, size_bytes) tuple.
    """
    dptr = ctypes.c_uint64(0)
    size = ctypes.c_size_t(0)
    result = _libcuda.cuModuleGetGlobal_v2(
        ctypes.byref(dptr),
        ctypes.byref(size),
        ctypes.c_void_p(cu_module),
        name.encode("utf-8"),
    )
    _check(result, f"cuModuleGetGlobal({name})")
    return dptr.value, size.value


def cuMemcpyHtoD(dst_device: int, src_host: ctypes.c_void_p, size: int):
    """Copy data from host to device.

    Args:
        dst_device: Device pointer (CUdeviceptr).
        src_host: Pointer to host data.
        size: Number of bytes to copy.
    """
    result = _libcuda.cuMemcpyHtoD_v2(
        ctypes.c_uint64(dst_device),
        src_host,
        ctypes.c_size_t(size),
    )
    _check(result, "cuMemcpyHtoD")


def write_device_global_ptr(cu_module: int, name: str, device_ptr: int):
    """Write a device pointer value into a module's global variable.

    This is the core operation: given a CUmodule from a Triton JIT-compiled kernel,
    set the module's copy of a device global (like g_prlx_buffer) to point to our
    allocated buffer.

    Args:
        cu_module: CUmodule handle.
        name: Name of the device global (e.g. "g_prlx_buffer").
        device_ptr: The device pointer value to write.
    """
    dptr, size = cuModuleGetGlobal(cu_module, name)
    # The global is a pointer, 8 bytes on 64-bit
    val = ctypes.c_uint64(device_ptr)
    cuMemcpyHtoD(dptr, ctypes.byref(val), 8)


def write_device_global_u32(cu_module: int, name: str, value: int):
    """Write a uint32 value into a module's global variable.

    Args:
        cu_module: CUmodule handle.
        name: Name of the device global (e.g. "__prlx_sample_rate").
        value: The uint32 value to write.
    """
    dptr, size = cuModuleGetGlobal(cu_module, name)
    val = ctypes.c_uint32(value)
    cuMemcpyHtoD(dptr, ctypes.byref(val), 4)
