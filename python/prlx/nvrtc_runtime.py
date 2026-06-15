"""
NVRTC runtime backend for PRLX: frictionless deep recording with no system
CUDA toolkit.

This is the pip-only port of the device recording runtime (the host side of
``lib/runtime/prlx_host.cu`` + the device record functions). It JIT-compiles the
recording functions together with the user's kernel via NVRTC (CuPy), sets the
``__device__`` globals from the host, launches, and reads the results back.

It supports two record modes:
  * FULL (mode 1): per-warp ring-buffer of 16-byte events (full detail).
  * FINGERPRINT (mode 2): a per-warp ``mix32`` rolling hash (divergence-proportional
    fast scan).

The LLVM pass inserts ``__prlx_record_branch`` calls; this runtime provides their
implementation + lifecycle. CuPy is imported lazily so importing PRLX never
requires a GPU.
"""

import struct

MODE_OFF, MODE_FULL, MODE_FP = 0, 1, 2

# Device-side recording functions + globals, compiled by NVRTC alongside the kernel.
PRLX_DEVICE_SOURCE = r'''
extern "C" {
__device__ unsigned long long g_prlx_buffer;       // full-capture ring buffers (device ptr)
__device__ unsigned long long g_prlx_fp_buffer;     // per-warp fingerprints (device ptr)
__device__ unsigned g_prlx_events_per_warp;
__device__ unsigned g_prlx_mode;                    // 0 off, 1 full, 2 fingerprint
}
__device__ __forceinline__ unsigned __prlx_tid() {
    return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
}
__device__ __forceinline__ unsigned __prlx_lane_id() { return __prlx_tid() & 31u; }
__device__ __forceinline__ unsigned __prlx_warp_id() {
    unsigned wib = __prlx_tid() >> 5;
    unsigned lb  = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    unsigned wpb = (blockDim.x*blockDim.y*blockDim.z + 31u) >> 5;
    return lb*wpb + wib;
}
extern "C" __device__ void __prlx_record_branch(unsigned site, unsigned cond, unsigned val) {
    if (g_prlx_mode == 0u) return;
    unsigned am = __activemask();              // all active lanes participate (convergent)
    if (__prlx_lane_id() != 0u) return;        // one record per warp
    unsigned warp = __prlx_warp_id();
    unsigned bd = cond & 1u;
    if (g_prlx_mode == 2u) {                    // FINGERPRINT: mix32, each warp owns its slot
        unsigned long long* fpb = (unsigned long long*)g_prlx_fp_buffer;
        unsigned h = (unsigned)fpb[warp];
        h = (h ^ site) * 2654435761u;
        h = (h ^ bd)   * 2654435761u;
        h = (h ^ am)   * 2654435761u;
        h = (h ^ val)  * 2654435761u;
        fpb[warp] = (unsigned long long)h;
    } else {                                    // FULL: ring-buffer event
        unsigned EPW = g_prlx_events_per_warp;
        unsigned* base = (unsigned*)g_prlx_buffer + (size_t)warp * (4u + EPW*4u);
        unsigned ix = atomicAdd(&base[0], 1u);
        if (ix < EPW) { unsigned* e = base + 4u + ix*4u; e[0]=site; e[1]=bd; e[2]=am; e[3]=val; }
        else atomicAdd(&base[1], 1u);
    }
}
'''

FP_SEED = 2166136261  # FNV offset basis; fingerprint buffer is pre-seeded with this


class PrlxNvrtcRuntime:
    """JIT-compile a kernel with PRLX recording and run it in FULL or FINGERPRINT mode.

    Usage::

        rt = PrlxNvrtcRuntime(events_per_warp=256).compile(kernel_src, "my_kernel")
        fp = rt.run_fingerprint(grid, block, (in_a, ...), num_warps)   # cheap scan
        full = rt.run_full(grid, block, (in_a, ...), num_warps)        # full detail
    """

    def __init__(self, events_per_warp=256):
        self.epw = int(events_per_warp)
        self._cp = None
        self._np = None
        self.mod = None
        self.fn = None

    def _lazy(self):
        if self._cp is None:
            import cupy as cp
            import numpy as np
            self._cp, self._np = cp, np
        return self._cp, self._np

    def compile(self, kernel_source, name, options=()):
        cp, _ = self._lazy()
        src = PRLX_DEVICE_SOURCE + "\n" + kernel_source
        self.mod = cp.RawModule(code=src, options=tuple(options))
        self.fn = self.mod.get_function(name)
        return self

    # set __device__ globals from the host (driver-API equivalent of cudaMemcpyToSymbol)
    def _set_global(self, gname, fmt, value):
        cp, np = self._cp, self._np
        g = self.mod.get_global(gname)
        host = np.frombuffer(struct.pack(fmt, int(value)), dtype=np.uint8)
        cp.cuda.runtime.memcpy(int(g), host.ctypes.data, host.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)

    def _set_u64(self, n, v): self._set_global(n, "<Q", v)
    def _set_u32(self, n, v): self._set_global(n, "<I", v)

    def stride(self):
        return 4 + self.epw * 4

    def run_fingerprint(self, grid, block, kernel_args, num_warps):
        cp, _ = self._lazy()
        fp = cp.full(num_warps, FP_SEED, dtype=cp.uint64)
        self._set_u64("g_prlx_fp_buffer", int(fp.data.ptr))
        self._set_u64("g_prlx_buffer", 0)
        self._set_u32("g_prlx_events_per_warp", self.epw)
        self._set_u32("g_prlx_mode", MODE_FP)
        self.fn(grid, block, kernel_args)
        return fp

    def run_full(self, grid, block, kernel_args, num_warps):
        cp, _ = self._lazy()
        buf = cp.zeros(num_warps * self.stride(), dtype=cp.uint32)
        self._set_u64("g_prlx_buffer", int(buf.data.ptr))
        self._set_u64("g_prlx_fp_buffer", 0)
        self._set_u32("g_prlx_events_per_warp", self.epw)
        self._set_u32("g_prlx_mode", MODE_FULL)
        self.fn(grid, block, kernel_args)
        return buf

    def events(self, full_buf, num_warps):
        """View a FULL buffer as [num_warps, events_per_warp, 4] (site, bd, mask, val)."""
        return full_buf.reshape(num_warps, self.stride())[:, 4:].reshape(num_warps, self.epw, 4)
