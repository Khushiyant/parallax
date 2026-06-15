# PRLX Architecture

## Overview

PRLX is a differential debugger for GPU kernels. It answers the question:
*"I ran my kernel twice with different inputs, where exactly did execution diverge, and why?"*

It works by instrumenting GPU code at compile time, recording per-warp
execution traces at runtime, and then diffing two traces offline.

## Data Flow

```
                         COMPILE TIME
                         ============

  Source code                     Triton kernel
  (CUDA C)                       (Python)
      |                               |
      v                               v
  clang -fpass-plugin=          prlx.enable() hooks
  libPrlxPass.so                stages["llir"]
      |                               |
      +--------> LLVM IR <------------+
                 (NVPTX / AMDGPU)
                    |
                    v
             PrlxPass walks IR:
             - finds BranchInst, CmpInst, AtomicRMW, StoreInst
             - inserts calls to __prlx_record_*
             - builds site table (hash -> source location)
                    |
                    v
             Instrumented binary
             + prlx-sites.json


                          RUNTIME
                          =======

  Instrumented kernel launches on GPU
      |
      v
  prlx_pre_launch()                  <-- host hook
      - allocates GPU buffers:
        trace (events), history, snapshot
      - writes TraceFileHeader to device memory
      - sets g_prlx_buffer device global
      |
      v
  Kernel runs, instrumented code calls:
      __prlx_record_branch(site_id, condition, operand)
      __prlx_record_snapshot(site_id, lhs, rhs, predicate)
      __prlx_record_value(site_id, value)
      |
      +-- Each call: atomicAdd(&write_idx, 1)
      |   writes 16-byte TraceEvent to per-warp ring buffer
      |   (overflow: increments overflow_count, drops event)
      v
  prlx_post_launch()                 <-- host hook
      - cudaDeviceSynchronize()
      - cudaMemcpy device -> host
      - writes .prlx file (optionally zstd-compressed)
      - frees GPU buffers


                         ANALYSIS
                         ========

  a.prlx    b.prlx
      \       /
       v     v
    prlx-diff (Rust)
        |
        +-- mmap both files
        +-- parse headers, validate kernel match
        +-- for each warp (parallel via rayon):
        |     align event streams with bounded lookahead
        |     classify divergences:
        |       Branch (dir_a != dir_b)
        |       ActiveMask (mask_a != mask_b)
        |       Value (val_a != val_b, opt-in)
        |       Path (site_a != site_b, no resync found)
        |       ExtraEvents (resync found, N events skipped)
        |     attach snapshot context if available
        |
        +-- output modes:
            - summary + per-site divergence report
            - interactive TUI (ratatui) with source view
            - JSON (for CI: prlx assert)
            - Chrome Trace Format (for flamegraph visualization)
```

## Components

### 1. LLVM Pass (`lib/pass/`)

A MODULE-level LLVM plugin loaded via `-fpass-plugin`. It operates on
NVPTX or AMDGPU IR (the intermediate representation for GPU device code).

**Target detection:**
- Uses `GPUTarget` enum (`None`, `NVPTX`, `AMDGPU`) to identify the module target
- NVPTX: detected by `"nvptx"` in target triple
- AMDGPU: detected by `"amdgcn"` in target triple
- Non-GPU modules are skipped entirely

**What it instruments:**
- `BranchInst` (conditional) -> `__prlx_record_branch`
- `ICmpInst` / `FCmpInst` used as predicates -> `__prlx_record_snapshot`
- `AtomicRMWInst` -> `__prlx_record_atomic`
- `StoreInst` to shared memory (addrspace 3) -> `__prlx_record_shmem_store`

**What it skips:**
- Functions named `__prlx_*` (prevents infinite recursion)
- NVIDIA builtins (`atomicAdd`, `__activemask`, `__shfl_sync`, etc.)
- AMD builtins (`__builtin_amdgcn_*`, `__ockl_*`, `__hip_atomic*`)
- Functions not matching `PRLX_FILTER` patterns (if set)

**Address space handling:**
- Shared memory: addrspace 3 on both NVPTX and AMDGPU
- Global memory: NVPTX uses AS 0 and 1; AMDGPU uses AS 1 only (AS 0 is flat/generic)

**Key design decisions:**
- Uses `getOrDeclare()` to avoid LLVM auto-renaming duplicate declarations
- Must NOT link LLVM static libs (LLVMCore, etc.), causes symbol conflicts
- Detects Triton's branchless kernels: `icmp` feeding inline PTX asm
  predicates (`@$2 ld.global.b32`) is treated as equivalent to a branch

**Output:** instrumented IR + `prlx-sites.json` mapping site_id hashes
to source locations.

### 2. Runtime (`lib/runtime/`)

Two halves: device-side recording functions and host-side lifecycle hooks.
Separate implementations for CUDA and HIP (AMD ROCm).

**Device side, CUDA** (`prlx_runtime.cu`):
- Recording functions called by instrumented code
- Each function: computes warp index, atomically claims a slot in the
  per-warp ring buffer, writes a 16-byte `TraceEvent`
- Overflow: if `write_idx >= events_per_warp`, increments `overflow_count`
  and returns (no crash, just drops events)
- Snapshot recording: uses `__shfl_sync(0xFFFFFFFF, value, lane)` to
  collect both comparison operands from all 32 lanes without branches
- History recording: writes to a separate ring buffer for time-travel context

**Device side, HIP** (`prlx_runtime_hip.cpp`):
- Same recording functions with AMD intrinsic replacements:
  - `__activemask()` -> `__ballot(1)`
  - `__shfl_sync(mask, val, src)` -> `__shfl(val, src)`
  - Lane ID: `__builtin_amdgcn_mbcnt_lo(~0u, 0)` (instead of PTX inline asm)
  - Cache-bypass stores: volatile stores (instead of PTX `st.global.cg`)
- Targets wave32 (RDNA GPUs). Wave64 (CDNA) deferred.

**Host side, CUDA** (`prlx_host.cu`):
- `prlx_pre_launch()`: allocates GPU memory for all buffers, writes header,
  sets device globals via `cudaMemcpyToSymbol`
- `prlx_post_launch()`: syncs device, copies buffers to host, writes `.prlx`
  file, frees GPU memory
- Session API: captures multiple kernel launches into a directory with manifest

**Host side, HIP** (`prlx_host_hip.cpp`):
- Mirrors CUDA host side with `hip*` API calls (`hipMalloc`, `hipMemcpy`,
  `hipMemcpyToSymbol(HIP_SYMBOL(...))`, `hipFree`, etc.)

**Platform detection** (`prlx_runtime.h`):
- `#if defined(__HIP_PLATFORM_AMD__)` selects HIP headers; otherwise CUDA headers
- Common `__device__` declarations guarded by `#if defined(__CUDACC__) || defined(__HIP__)`

### 3. Differ (`differ/`, Rust)

Compares two `.prlx` trace files and identifies divergences.

**Alignment algorithm** (`differ.rs`):
- For each warp, walks both event streams in lockstep
- When events match (same site_id): compare branch_dir, active_mask,
  optionally value_a
- When events mismatch (different site_id): bounded lookahead search
  - Scan up to `lookahead_window` (default 32) events ahead in both traces
  - If resync point found: report `ExtraEvents` and resume
  - If not found within window: report `Path` divergence

**Session mode** (`--session`):
- Compares two session directories (each containing a `session.json` manifest
  and per-kernel `.prlx` files)
- Matches launches by kernel name + launch index
- Reports unmatched launches in either session (kernels in A but not B, or vice versa)
- Warns on grid/block dimension mismatches between matching launches

**Output modes:**
- Summary: "N divergences across M warps at K sites"
- Detailed: per-site, per-warp, per-event divergence report with optional
  per-lane snapshot rendering
- TUI: interactive terminal UI (ratatui + crossterm) with source view
- JSON (`--json`): structured output for CI integration (`json_output.rs`)
- Flamegraph (`--export-flamegraph`): Chrome Trace Format (`flamegraph.rs`)

**JSON output** (`json_output.rs`):
- `JsonDiffReport` with status, divergence counts, threshold, passed bool
- `ignore_active_mask` filtering: count only branch/path/value divergences
- Used by `prlx assert` for CI regression gating

**Flamegraph export** (`flamegraph.rs`):
- Chrome Trace Format (catapult JSON) for `chrome://tracing` or Perfetto
- pid = CUDA/HIP block index, tid = warp within block
- Duration events (`ph: "X"`) per divergence, counter events (`ph: "C"`)
  for per-site frequency heatmap

**TUI source view** (`tui/source_cache.rs`):
- Press `s` to toggle inline source code at the selected divergence site
- `SourceCache` lazily loads files, caches as `None` for unreadable files
- Target line highlighted in yellow with `>` marker
- Requires `--map` for site-to-source mapping

### 4. Python Package (`python/prlx/`)

User-facing interface, Triton integration, and PyTorch integration.

- `prlx.enable()` -> installs Triton compiler hook
- `prlx.enable_pytorch()` -> installs PyTorch hooks (three tiers)
- `prlx.pytorch_trace(name)` -> context manager for session tracing
- `prlx.read_trace(path)` -> pure-Python trace parser (struct + mmap)
- `prlx.diff_traces(a, b)` -> shells out to `prlx-diff`
- `prlx compile` CLI -> wraps clang with the pass plugin
- `prlx diff` CLI -> wraps `prlx-diff` with auto site-map discovery
- `prlx assert` CLI -> CI regression gate with threshold-based pass/fail
- `prlx flamegraph` CLI -> Chrome Trace Format export
- `prlx pytorch` CLI -> run scripts with PyTorch instrumentation
- `prlx session capture/inspect/diff` CLI -> multi-kernel session management

**Triton hook** (`triton_hook.py`):
- Intercepts `stages["llir"]` via `knobs.runtime.add_stages_inspection_hook`
- Pipeline: llvm-link (merge runtime BC) -> opt (run pass) -> return IR string
- Handles Triton's extended NVPTX data layout with `--suppress-warnings`

**PyTorch hook** (`pytorch_hook.py`):
Three-tier instrumentation for PyTorch workloads:

- **Tier 1, Triton via torch.compile**: Delegates to `triton_hook.install()`.
  Instruments kernels compiled through `torch.compile` / Inductor.
- **Tier 2, load_inline hook**: Monkey-patches
  `torch.utils.cpp_extension.load_inline` to inject `-fpass-plugin=libPrlxPass.so`
  into `extra_cuda_cflags` and link the runtime library via `extra_ldflags`.
  Only works when extension code is clang-compatible.
- **Tier 3, NVBit fallback**: Sets `LD_PRELOAD=libprlx_nvbit.so` for
  SASS-level binary instrumentation of pre-compiled CUDA ops. Must be
  configured before CUDA context creation.

`PrlxTorchWrapper` is a context manager that sets `PRLX_SESSION` and
`PRLX_TRACE` environment variables for scoped session tracing.

`uninstall()` cleanly reverses all three tiers.

### 5. NVBit Tool (`lib/nvbit_tool/`) _(experimental)_

SASS-level binary instrumentation for closed-source kernels.
Uses NVBit to inject instrumentation at the GPU assembly level.
Less tested than the LLVM pass path.

### 6. Build System (`CMakeLists.txt`)

Conditional multi-platform build:

- `PRLX_ENABLE_CUDA` (default ON): Builds CUDA runtime (`prlx_runtime`,
  `prlx_runtime_shared`), NVPTX bitcode, and examples
- `PRLX_ENABLE_HIP` (default OFF): Builds HIP runtime (`prlx_runtime_hip`)
  with `find_package(hip)` and ROCm SDK detection
- The LLVM pass (`libPrlxPass.so`) is always built, it handles both NVPTX
  and AMDGPU targets at runtime via `getGPUTarget()`
- The differ (`prlx-diff`) is target-agnostic, trace format is the same
  regardless of GPU vendor

## Trace File Format (`.prlx`)

```
Offset  Size    Field
──────  ──────  ─────
0       160B    TraceFileHeader
                  magic (8B): 0x50524C5800000000 ("PRLX")
                  version (4B), flags (4B)
                  kernel_name_hash (8B), kernel_name (64B)
                  grid_dim (12B), block_dim (12B)
                  warps_per_block (4B), total_warp_slots (4B)
                  events_per_warp (4B)
                  [4B padding]
                  timestamp (8B), cuda_arch (4B)
                  history_depth (4B), history_section_offset (4B)
                  sample_rate (4B)
                  snapshot_depth (4B), snapshot_section_offset (4B)

160     N*WBS   Warp Event Buffers (one per warp)
                  WarpBufferHeader (16B):
                    write_idx, overflow_count, num_events, total_event_count
                  TraceEvent[events_per_warp] (events_per_warp * 16B):
                    site_id (4B), event_type (1B), branch_dir (1B),
                    _reserved (2B), active_mask (4B), value_a (4B)

WBS = 16 + events_per_warp * 16   (default: 65552 bytes)

---  If PRLX_FLAG_HISTORY set:  ---

H_OFF   N*HRS   History Ring Buffers (one per warp)
                  HistoryRingHeader (16B):
                    write_idx, depth, total_writes, _reserved
                  HistoryEntry[depth] (depth * 16B):
                    site_id (4B), value (4B), seq (4B), _pad (4B)

HRS = 16 + history_depth * 16     (default depth=64: 1040 bytes)

---  If PRLX_FLAG_SNAPSHOT set:  ---

S_OFF   N*SRS   Snapshot Ring Buffers (one per warp)
                  SnapshotRingHeader (16B):
                    write_idx, depth, total_writes, _reserved
                  SnapshotEntry[depth] (depth * 288B):
                    site_id (4B), active_mask (4B), seq (4B),
                    cmp_predicate (4B),
                    lhs_values[32] (128B), rhs_values[32] (128B),
                    _pad (16B)

SRS = 16 + snapshot_depth * 288    (default depth=32: 9232 bytes)
```

All multi-byte fields are little-endian. Structures are 16-byte aligned
for `v4.u32` PTX store compatibility.

## Scalability Math

PRLX is a debugger, not a production profiler. You use it to find a bug,
then turn it off. Still, it's worth knowing the resource cost so you can
size your debugging sessions appropriately.

### GPU Memory (allocated by `prlx_pre_launch`)

```
total_warps = grid_blocks * ceil(threads_per_block / 32)

Event buffer  = total_warps * (16 + events_per_warp * 16) bytes
History buffer = total_warps * (16 + history_depth * 16) bytes
Snapshot buffer = total_warps * (16 + snapshot_depth * 288) bytes
```

With defaults (`events_per_warp=4096`, `history_depth=64`, `snapshot_depth=32`):

| Per-warp cost     | Size     |
|-------------------|----------|
| Events            | 64 KB    |
| History           | 1 KB     |
| Snapshots         | 9 KB     |
| **Total**         | **~74 KB** |

Scaling by kernel size:

| Kernel config                 | Warps  | Events only | All features |
|-------------------------------|--------|-------------|--------------|
| 1 block, 32 threads          | 1      | 64 KB       | 74 KB        |
| 16 blocks, 256 threads       | 128    | 8 MB        | 9.5 MB       |
| 128 blocks, 256 threads      | 1,024  | 64 MB       | 74 MB        |
| 512 blocks, 512 threads      | 8,192  | 512 MB      | 592 MB       |
| 1024 blocks, 1024 threads    | 32,768 | 2 GB        | 2.3 GB       |

**Rule of thumb:** ~64 KB per warp for events alone. If your kernel has
more warps than your GPU has MB of free VRAM, use `PRLX_SAMPLE_RATE`
to reduce event density.

### Trace File Size (on disk)

The file size equals the GPU allocation (header + event buffers + optional
sections). A 1024-warp kernel produces a ~64 MB trace by default.

With `PRLX_COMPRESS=1` (zstd), expect 5-15x reduction depending on event
density. Sparse traces (most warps idle) compress very well.

### Sampling Trade-off

`PRLX_SAMPLE_RATE=N` records 1 in every N events. This does NOT reduce
buffer allocation, buffers are pre-allocated at full size. What it does:
- Reduces `write_idx` advancement rate -> fewer overflow drops
- Makes traces sparser -> better compression ratio
- Loses fine-grained event ordering within a warp

For large kernels where you'd overflow the 4096-event buffer, sampling
at 4x-16x lets you still capture the overall control flow shape.

### Overflow Behavior

When a warp generates more than `events_per_warp` events:
- `overflow_count` is incremented atomically
- The event is silently dropped (no crash, no corruption)
- The differ reports overflow counts in the summary
- Use `PRLX_SAMPLE_RATE` to mitigate

### Practical Guidance

| Scenario                          | Recommended config                            |
|-----------------------------------|-----------------------------------------------|
| Small kernel (<100 warps)         | Defaults. Turn on snapshots: `PRLX_SNAPSHOT_DEPTH=32` |
| Medium kernel (100-1K warps)      | Defaults work. ~64-640 MB VRAM.               |
| Large kernel (1K-10K warps)       | Sample: `PRLX_SAMPLE_RATE=4`                  |
| Very large kernel (>10K warps)    | Sample: `PRLX_SAMPLE_RATE=8` or higher        |
| Production binary (no recompile)  | NVBit backend (experimental)                  |
| Triton kernel                     | `prlx.enable()`, same env vars apply          |
| PyTorch model                     | `prlx.enable_pytorch()` (Triton + load_inline + NVBit) |
| AMD GPU (RDNA, wave32)            | Build with `-DPRLX_ENABLE_HIP=ON`             |

