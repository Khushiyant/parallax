"""Binds PRLX device globals into Triton JIT-compiled GPU modules.

The core problem: cudaMemcpyToSymbol / hipMemcpyToSymbol in prlx_host writes
device globals (g_prlx_buffer, etc.) into the shared library's module, but
Triton kernels are loaded into separate modules via cuModuleLoadData /
hipModuleLoadData. Their copies of these globals stay nullptr, so recording
functions bail out and capture 0 events.

This module uses the CUDA Driver API or HIP API to write the correct buffer
pointers into each Triton kernel's module before launch.  Backend is
auto-detected: CUDA first, then HIP.
"""

from typing import Optional, Set

from ._log import get_logger
from .runtime import PrlxRuntime

logger = get_logger(__name__)

_BACKEND: Optional[str] = None


def _detect_backend() -> Optional[str]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    from . import _cuda_driver as cu
    if cu.available():
        _BACKEND = "cuda"
        logger.debug("backend: CUDA")
        return _BACKEND

    from . import _hip_driver as hip
    if hip.available():
        _BACKEND = "hip"
        logger.debug("backend: HIP")
        return _BACKEND

    return None


class ModuleBinder:
    """Binds PRLX runtime state into Triton kernel GPU modules."""

    def __init__(self, runtime: PrlxRuntime, verbose: bool = False):
        self._runtime = runtime
        self._verbose = verbose
        self._bound_modules: Set[int] = set()
        self._backend = _detect_backend()

    def invalidate(self):
        """Clear the bound-modules cache.

        Must be called after pre_launch() because buffer pointers change
        between launches (freed in post_launch, reallocated in pre_launch).
        """
        self._bound_modules.clear()

    def bind_module(self, cu_function: int):
        """Write all PRLX device globals into the module owning *cu_function*.

        Args:
            cu_function: CUfunction/hipFunction_t handle from launch hook metadata.
        """
        if self._backend == "cuda":
            self._bind_cuda(cu_function)
        elif self._backend == "hip":
            self._bind_hip(cu_function)

    def _bind_cuda(self, cu_function: int):
        from . import _cuda_driver as cu

        cu_module = cu.cuFuncGetModule(cu_function)
        if cu_module in self._bound_modules:
            return

        trace_buf = self._runtime.get_trace_buffer()
        if trace_buf == 0:
            return

        logger.debug("Binding device globals to CUmodule 0x%x", cu_module)

        cu.write_device_global_ptr(cu_module, "g_prlx_buffer", trace_buf)
        cu.write_device_global_ptr(cu_module, "g_prlx_history_buffer", self._runtime.get_history_buffer())
        cu.write_device_global_ptr(cu_module, "g_prlx_snapshot_buffer", self._runtime.get_snapshot_buffer())
        cu.write_device_global_u32(cu_module, "__prlx_sample_rate", self._runtime.get_sample_rate())
        cu.write_device_global_u32(cu_module, "g_prlx_history_depth", self._runtime.get_history_depth())
        cu.write_device_global_u32(cu_module, "g_prlx_snapshot_depth", self._runtime.get_snapshot_depth())

        self._bound_modules.add(cu_module)

    def _bind_hip(self, hip_function: int):
        from . import _hip_driver as hip

        # HIP does not expose hipFuncGetModule; the function handle IS the
        # module in Triton's HIP backend, pass it directly.
        hip_module = hip_function
        if hip_module in self._bound_modules:
            return

        trace_buf = self._runtime.get_trace_buffer()
        if trace_buf == 0:
            return

        logger.debug("Binding device globals to hipModule 0x%x", hip_module)

        hip.write_device_global_ptr(hip_module, "g_prlx_buffer", trace_buf)
        hip.write_device_global_ptr(hip_module, "g_prlx_history_buffer", self._runtime.get_history_buffer())
        hip.write_device_global_ptr(hip_module, "g_prlx_snapshot_buffer", self._runtime.get_snapshot_buffer())
        hip.write_device_global_u32(hip_module, "__prlx_sample_rate", self._runtime.get_sample_rate())
        hip.write_device_global_u32(hip_module, "g_prlx_history_depth", self._runtime.get_history_depth())
        hip.write_device_global_u32(hip_module, "g_prlx_snapshot_depth", self._runtime.get_snapshot_depth())

        self._bound_modules.add(hip_module)
