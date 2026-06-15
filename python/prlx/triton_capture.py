"""
Trace capture for Triton kernels: the host side that arms PRLX's device globals
on a Triton-loaded module so the instrumented kernel actually records.

Flow:
  1. Compile the kernel once (PRLX Triton hook instruments it). The launch returns
     a Triton ``CompiledKernel``.
  2. ``arm(compiled, total_warps)`` allocates a trace buffer and points the module's
     ``g_prlx_buffer`` global at it (driver-API ``cuModuleGetGlobal`` + memcpy).
  3. Relaunch the kernel (same cached module); the record functions write events.
  4. ``events_per_warp_counts(buf, total_warps)`` reads how many events each warp wrote.

CuPy is imported lazily so importing PRLX never requires a GPU.
"""

import struct

HEADER_SIZE = 160          # TraceFileHeader
WARP_HEADER_SIZE = 16      # WarpBufferHeader (write_idx is field 0)
EVENT_SIZE = 16            # TraceEvent
DEFAULT_EVENTS_PER_WARP = 4096   # PRLX_EVENTS_PER_WARP (compile-time constant in the runtime)

SNAP_RING_HEADER = 16      # SnapshotRingHeader (write_idx, depth, total_writes, _reserved)
SNAP_ENTRY = 288           # SnapshotEntry (site, mask, seq, pred, lhs[32], rhs[32], pad)
HEADER_FMT = "<QII Q 64s 3I 3I II I 4x Q I II 3I"
MAGIC = 0x50524C5800000000
VERSION = 1
FLAG_SNAPSHOT = 0x10


def _snap_ring_bytes(depth):
    return SNAP_RING_HEADER + depth * SNAP_ENTRY


def _warp_buf_bytes(events_per_warp):
    return WARP_HEADER_SIZE + events_per_warp * EVENT_SIZE


def buffer_words(total_warps, events_per_warp=DEFAULT_EVENTS_PER_WARP):
    return (HEADER_SIZE + total_warps * _warp_buf_bytes(events_per_warp)) // 4


def total_warps_for(grid_blocks, num_warps):
    """Warps = grid blocks x warps/block (Triton block = num_warps*32 threads)."""
    return int(grid_blocks) * int(num_warps)


def arm(compiled_kernel, total_warps, events_per_warp=DEFAULT_EVENTS_PER_WARP, global_name="g_prlx_buffer"):
    """Allocate a trace buffer and point the Triton module's g_prlx_buffer at it.

    Call AFTER the first (compiling) launch, BEFORE the recording launch.
    Returns the cupy buffer to read with :func:`events_per_warp_counts` after relaunch.
    """
    import cupy as cp
    import numpy as np
    from cupy.cuda import driver

    buf = cp.zeros(buffer_words(total_warps, events_per_warp), dtype=cp.uint32)
    gptr = driver.moduleGetGlobal(compiled_kernel.module, global_name)
    addr = np.array([int(buf.data.ptr)], dtype=np.uint64)
    cp.cuda.runtime.memcpy(int(gptr), addr.ctypes.data, 8, cp.cuda.runtime.memcpyHostToDevice)
    return buf


def events_per_warp_counts(buf, total_warps, events_per_warp=DEFAULT_EVENTS_PER_WARP):
    """Return the number of events each warp recorded (WarpBufferHeader.write_idx)."""
    import cupy as cp
    import numpy as np

    host = cp.asnumpy(buf).view(np.uint32)
    wb_words = _warp_buf_bytes(events_per_warp) // 4
    base = HEADER_SIZE // 4
    return [int(host[base + w * wb_words]) for w in range(total_warps)]


def total_events(buf, total_warps, events_per_warp=DEFAULT_EVENTS_PER_WARP):
    return sum(events_per_warp_counts(buf, total_warps, events_per_warp))


def arm_snapshots(compiled_kernel, total_warps, depth=32):
    """Arm per-lane operand snapshot capture: point g_prlx_snapshot_buffer at a
    fresh buffer and set g_prlx_snapshot_depth. Call alongside :func:`arm`."""
    import cupy as cp
    import numpy as np
    from cupy.cuda import driver

    snap = cp.zeros((total_warps * _snap_ring_bytes(depth)) // 4, dtype=cp.uint32)
    gbuf = driver.moduleGetGlobal(compiled_kernel.module, "g_prlx_snapshot_buffer")
    gdep = driver.moduleGetGlobal(compiled_kernel.module, "g_prlx_snapshot_depth")
    addr = np.array([int(snap.data.ptr)], dtype=np.uint64)
    cp.cuda.runtime.memcpy(int(gbuf), addr.ctypes.data, 8, cp.cuda.runtime.memcpyHostToDevice)
    dep = np.array([depth], dtype=np.uint32)
    cp.cuda.runtime.memcpy(int(gdep), dep.ctypes.data, 4, cp.cuda.runtime.memcpyHostToDevice)
    return snap


def snapshot_writes(snap_buf, total_warps, depth):
    """Per-warp snapshot write counts (SnapshotRingHeader.write_idx)."""
    import cupy as cp
    import numpy as np

    host = cp.asnumpy(snap_buf).view(np.uint32)
    ring_words = _snap_ring_bytes(depth) // 4
    return [int(host[w * ring_words]) for w in range(total_warps)]


def write_prlx(path, event_buf, total_warps, grid_blocks, num_warps,
               events_per_warp=DEFAULT_EVENTS_PER_WARP,
               snapshot_buf=None, snapshot_depth=0, kernel_name="triton_kernel", cuda_arch=120):
    """Assemble a valid .prlx file from a captured event buffer (+ optional snapshots)."""
    import cupy as cp

    warp_buf = _warp_buf_bytes(events_per_warp)
    raw = cp.asnumpy(event_buf).tobytes()
    ev = bytearray(raw[HEADER_SIZE:HEADER_SIZE + total_warps * warp_buf])
    for w in range(total_warps):                       # set num_events = min(write_idx, EPW)
        base = w * warp_buf
        wi = struct.unpack_from("<I", ev, base)[0]
        struct.pack_into("<I", ev, base + 8, min(wi, events_per_warp))

    snap_section = b""
    snap_off = 0
    flags = 0
    if snapshot_buf is not None and snapshot_depth > 0:
        flags = FLAG_SNAPSHOT
        snap_off = HEADER_SIZE + total_warps * warp_buf       # no history section
        snap = bytearray(cp.asnumpy(snapshot_buf).tobytes())
        ring = _snap_ring_bytes(snapshot_depth)
        for w in range(total_warps):                          # set ring depth field for the reader
            struct.pack_into("<I", snap, w * ring + 4, snapshot_depth)
        snap_section = bytes(snap)

    name = kernel_name.encode()[:63]
    name = name + b"\x00" * (64 - len(name))
    hdr = struct.pack(HEADER_FMT, MAGIC, VERSION, flags, 0xABCD, name,
                      grid_blocks, 1, 1, num_warps * 32, 1, 1,
                      num_warps, total_warps, events_per_warp,
                      0, cuda_arch, 0, 0, 1, snapshot_depth, snap_off)
    assert len(hdr) == HEADER_SIZE
    from pathlib import Path
    Path(path).write_bytes(bytes(hdr) + bytes(ev) + snap_section)
