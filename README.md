# PRLX
<img width="1000" height="500" alt="prlx" src="https://github.com/user-attachments/assets/b3f2581b-28c1-4846-b3b6-1d63811d582b" />

Differential debugger for CUDA and Triton GPU kernels.

Most GPU tools tell you *that* a kernel went wrong. PRLX tells you *where two runs
diverged and why*. Run a kernel twice (two inputs, or a known-good version against
a buggy one) and PRLX instruments every branch, records a per-warp execution trace,
and diffs them: the exact warp that diverged, the instruction it diverged at, and
the operands every lane saw.

```
Site 0xfbe6edc1  (branch_kernel:12)  2 warps affected

  Warp 1, event 3: Branch Direction
    A: TAKEN    B: NOT-TAKEN

    Operand Snapshot (icmp sgt):
    Lane       A:lhs    A:rhs       B:lhs    B:rhs
       0          32       10          32       64  <<<
       1          33       10          33       64  <<<
       2          34       10          34       64  <<<
      ...
```

The threshold changed from 10 to 64. Lanes 0 to 31 compared their value against it;
in run A they all passed, in run B they didn't. That's the bug, found in one diff.

## Why PRLX

- **Run-vs-run, down to the lane.** Profilers show aggregate stats and `cuda-gdb`
  inspects one run. PRLX compares two executions at per-warp, per-instruction,
  per-lane operand granularity.
- **CUDA, Triton, and PyTorch.** Instruments hand-written CUDA C, Triton kernels
  (via the compiler hook, no kernel changes), and `torch.compile` / inline extensions.
- **CI-ready.** Gate a build on "did this kernel start behaving differently than the
  golden run" with a single `prlx assert`.
- **NVIDIA and AMD.** One pass covers NVPTX and AMDGPU; one trace format covers both.

## Install

```bash
pip install prlx
```

Trace analysis works out of the box on Linux x86_64 with no other dependencies:
`prlx diff`, `prlx assert`, the TUI, flamegraph export, and the Python reader.

### Capturing traces from your own kernels

Instrumenting and recording kernels needs a few prerequisites on the machine:

- A **CUDA GPU**.
- **LLVM 18, 19, or 20** with `opt` and `llvm-link` on your `PATH`. On Ubuntu/Debian:

  ```bash
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh && sudo ./llvm.sh 20
  export PATH="/usr/lib/llvm-20/bin:$PATH"   # so opt / llvm-link are found
  opt --version                               # should report 20.x
  ```

- **Triton / PyTorch** kernels: `pip install "prlx[pytorch]"` (and `triton` if you use it directly).
- The hand-written **CUDA C** path (`prlx compile`): the **CUDA toolkit** (`nvcc`).

> LLVM is a manual prerequisite for capture today. Bundling it so `pip install` is
> fully self-contained is planned.

<details>
<summary>Build from source</summary>

Requires CUDA 12+, LLVM/Clang 18-20, Rust, and CMake 3.20+:

```bash
cmake -B build && cmake --build build     # LLVM pass + CUDA runtime
(cd differ && cargo build --release)       # the differ
pip install -e .
```
</details>

<details>
<summary>AMD ROCm</summary>

```bash
cmake -B build -DPRLX_ENABLE_CUDA=OFF -DPRLX_ENABLE_HIP=ON
cmake --build build
(cd differ && cargo build --release)
pip install -e .
```

Build deps: CMake 3.20+, LLVM/Clang 18-20, ROCm 5.0+, Rust stable.
The LLVM pass supports AMDGPU targets; the HIP runtime targets wave32 (RDNA) GPUs.
</details>

<details>
<summary>Prebuilt wheel</summary>

`scripts/build_wheel.sh` compiles the passes, runtime, and differ and bundles them
into a platform wheel, so installing it needs no build toolchain:

```bash
bash scripts/build_wheel.sh
pip install dist/*.whl
```

(The Triton path still expects `opt`/`llvm-link` on `PATH`.)
</details>

## Usage

### CUDA C

```bash
prlx compile kernel.cu -o kernel
PRLX_TRACE=a.prlx PRLX_SNAPSHOT_DEPTH=32 ./kernel --input-a
PRLX_TRACE=b.prlx PRLX_SNAPSHOT_DEPTH=32 ./kernel --input-b
prlx diff a.prlx b.prlx
```

### Triton

```python
import prlx
prlx.enable()  # hooks the Triton compiler; no kernel changes needed

import os, triton

os.environ["PRLX_TRACE"] = "a.prlx"
my_kernel[grid](...)

os.environ["PRLX_TRACE"] = "b.prlx"
my_kernel[grid](...)
```

### Python API

```python
from prlx import read_trace, diff_traces

# Read traces directly
trace = read_trace("a.prlx")
print(trace.header.kernel_name, trace.num_warps, "warps")
for w in trace.warps():
    for ev in w.events:
        if ev.is_branch:
            print(f"  warp {w.warp_idx}: site {ev.site_id:#x} {'T' if ev.branch_taken else 'NT'}")

# Or just run the differ
diff_traces("a.prlx", "b.prlx", history=True)
```

### Multi-Kernel Pipelines

Capture and diff entire GPU pipelines (multiple kernel launches):

```c
// In your code:
prlx_session_begin(NULL);
kernel_A<<<grid, block>>>(...);  // prlx_pre/post_launch called automatically
kernel_B<<<grid, block>>>(...);
prlx_session_end();
```

```bash
# Capture sessions
PRLX_SESSION=/tmp/session_a ./my_pipeline --param-a
PRLX_SESSION=/tmp/session_b ./my_pipeline --param-b

# Diff two sessions
prlx diff /tmp/session_a /tmp/session_b

# Or use the session subcommand:
prlx session diff /tmp/session_a /tmp/session_b

# Inspect a session manifest:
prlx session inspect /tmp/session_a

# Capture via CLI wrapper:
prlx session capture ./my_pipeline -o /tmp/session_a -- --param-a
```

Unmatched kernel launches between sessions are reported as warnings. Grid/block dimension mismatches are also flagged.

### PyTorch

```python
import prlx

# Hooks Triton (torch.compile) + load_inline (C++ extensions) automatically
prlx.enable_pytorch()

model = MyModel().cuda()
output = model(input_tensor)  # kernels are instrumented

# Or use the context manager for session tracing:
with prlx.pytorch_trace("my_model", output="/tmp/trace"):
    model(input_tensor)
```

```bash
# Run a script with PyTorch instrumentation
prlx pytorch run script.py

# NVBit fallback for pre-compiled ops (no recompilation needed)
prlx pytorch run --nvbit script.py

# Check integration status
prlx pytorch --info
```

Install the optional PyTorch dependency: `pip install prlx[pytorch]`

### TUI

```bash
prlx diff a.prlx b.prlx --tui --map prlx-sites.json
```

Interactive terminal UI for navigating divergences across warps. Press `s` to toggle inline source view at divergence sites (requires `--map` for site-to-source mapping).

| Key | Action |
|-----|--------|
| `j`/`k` | Scroll up/down |
| `n`/`N` | Next/previous divergence |
| `]`/`[` | Next/previous warp |
| `s` | Toggle source view |
| `Tab` | Switch pane focus |
| `/` | Jump to warp by number |
| `q` | Quit |

### CI Regression Gate

Automatically pass/fail based on divergence thresholds:

```bash
# Strict: zero divergences allowed (default)
prlx assert a.prlx b.prlx

# Tolerant: allow up to 5 divergences
prlx assert a.prlx b.prlx --max-divergences 5

# Golden mode: compare against a known-good trace
prlx assert --golden golden.prlx test.prlx

# JSON output for CI pipelines
prlx assert a.prlx b.prlx --json

# Ignore active mask differences (only count branch/path/value)
prlx assert a.prlx b.prlx --ignore-active-mask
```

Exit code 0 = pass, 1 = fail. Human-readable summary by default:

```
PRLX ASSERT: PASS (4 divergences, threshold: 5)
PRLX ASSERT: FAIL (4 divergences, threshold: 2)
```

### Flamegraph Export

Export divergences to Chrome Trace Format for visual analysis:

```bash
prlx flamegraph a.prlx b.prlx -o divergences.json --map prlx-sites.json
```

Open `divergences.json` in `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev). Each row is a warp (grouped by block), colored bars show divergence events, and counter tracks show per-site frequency heatmaps.

## Environment Variables

| Variable | Default | What it does |
|---|---|---|
| `PRLX_TRACE` | `trace.prlx` | Output path |
| `PRLX_SNAPSHOT_DEPTH` | `0` | Per-lane operand ring buffer size |
| `PRLX_HISTORY_DEPTH` | `0` | Time-travel value ring buffer size |
| `PRLX_SAMPLE_RATE` | `1` | Record 1 in N events |
| `PRLX_COMPRESS` | `0` | zstd compress the trace |
| `PRLX_ENABLED` | `1` | Kill switch |
| `PRLX_FILTER` | _(none)_ | Comma-separated glob patterns for kernel names to instrument |
| `PRLX_SESSION` | _(none)_ | Directory path for multi-launch session mode |
| `PRLX_SITES` | `prlx-sites.json` | Output path for site map |
| `PRLX_INSTRUMENT_STORES` | `0` | Instrument global memory stores (opt-in, produces large traces) |
| `PRLX_OPT_TIMEOUT` | `120` | Timeout (seconds) for llvm-link/opt in Triton hook |

## How It Works

PRLX has three backends for instrumenting GPU code:

1. **LLVM pass** (`lib/pass/`): loaded as `-fpass-plugin` during compilation (clang) or injected between Triton's `make_llir` and `make_ptx` stages. Walks NVPTX or AMDGPU IR and inserts calls to `__prlx_record_branch` / `__prlx_record_value` at every branch and comparison. For Triton's branchless single-BB kernels it detects predicated ops (`icmp` feeding inline PTX asm or `select`). Supports both NVIDIA (NVPTX) and AMD (AMDGPU) targets.

2. **NVBit tool** (`lib/nvbit_tool/`) _(experimental)_: SASS-level binary instrumentation via NVBit. Works on closed-source kernels where you don't have IR access. Less tested than the LLVM pass; use it when recompilation is not possible.

3. **Runtime** (`lib/runtime/`): device-side ring buffers (one per warp) that record events, value history, and per-lane comparison operand snapshots. Host hooks (`prlx_pre_launch` / `prlx_post_launch`) manage allocation and readback.

Traces are written to `.prlx` files (a compact binary format, optionally zstd-compressed). The **differ** (`differ/`, Rust) aligns event streams with bounded lookahead, classifies divergences (branch direction, path length, missing events), and can display per-lane operand diffs.

## Layout

```
lib/pass/           LLVM instrumentation pass (libPrlxPass.so), NVPTX + AMDGPU
lib/runtime/        device-side recording + host hooks (CUDA + HIP)
lib/nvbit_tool/     NVBit binary instrumentation backend (experimental)
lib/common/         shared trace format header
differ/             Rust differ + TUI + JSON/flamegraph export (prlx-diff)
python/prlx/        trace reader, Triton hook, PyTorch hook, runtime FFI, CLI
examples/           demo kernels (branch, loop, matmul, occupancy)
tools/              utilities (gen_demo_traces.py, synthetic trace generator)
```

## License

MIT
</content>
