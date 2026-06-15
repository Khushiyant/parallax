"""PRLX CLI, console entry point for the GPU differential debugger."""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import _find_lib

logger = logging.getLogger(__name__)

try:
    from importlib.metadata import version as _meta_version
    PRLX_VERSION = _meta_version("prlx")
except Exception as e:
    PRLX_VERSION = "dev"
    logger.debug("Could not read prlx version from metadata: %s", e)

PRLX_BANNER = r"""
 ____  ____  _     __  __
|  _ \|  _ \| |    \ \/ /
| |_) | |_) | |     \  /
|  __/|  _ <| |___  /  \
|_|   |_| \_\_____|/_/\_\  v{version}

PRLX, GPU Differential Debugger
{gpu_line}
"""


def _detect_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().split("\n")[0]
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            sm = parts[1].replace(".", "")
            return f"Auto-detected: NVIDIA {name} (SM_{sm})"
    except FileNotFoundError:
        logger.debug("nvidia-smi not found in PATH")
    except subprocess.SubprocessError as e:
        logger.debug("nvidia-smi failed: %s", e)
    return "No GPU detected"


def print_banner():
    gpu_info = _detect_gpu()
    banner = PRLX_BANNER.format(
        version=PRLX_VERSION,
        gpu_line=f"(c) 2026 | {gpu_info}",
    )
    print(banner, file=sys.stderr)


def find_site_map(search_dir=None):
    if search_dir is None:
        search_dir = Path.cwd()
    else:
        search_dir = Path(search_dir)

    candidates = [
        search_dir / "prlx-sites.json",
        search_dir / "build" / "prlx-sites.json",
        search_dir / ".." / "build" / "prlx-sites.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    current = search_dir
    for _ in range(5):
        site_map = current / "prlx-sites.json"
        if site_map.exists():
            return site_map.resolve()
        current = current.parent

    return None


def detect_gpu_arch():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(".")
                if len(parts) == 2:
                    major, minor = int(parts[0]), int(parts[1])
                    return major * 10 + minor
    except FileNotFoundError:
        logger.debug("nvidia-smi not found (detect_gpu_arch)")
    except subprocess.SubprocessError as e:
        logger.debug("nvidia-smi failed (detect_gpu_arch): %s", e)
    return None


def find_clang_for_pass(pass_lib):
    if pass_lib is None or not pass_lib.exists():
        return None, None

    try:
        result = subprocess.run(
            ["readelf", "-d", str(pass_lib)],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "libLLVM.so." in line:
                m = re.search(r"libLLVM\.so\.(\d+)", line)
                if m:
                    llvm_ver = int(m.group(1))
                    clang = f"clang++-{llvm_ver}"
                    clang_path = subprocess.run(
                        ["which", clang], capture_output=True, text=True,
                    )
                    if clang_path.returncode == 0:
                        return clang_path.stdout.strip(), llvm_ver
    except FileNotFoundError:
        logger.debug("readelf not found in PATH")
    except subprocess.SubprocessError as e:
        logger.debug("readelf failed: %s", e)

    # For bundled versioned pass (libPrlxPass.llvm20.so), extract from filename
    m = re.search(r"llvm(\d+)", pass_lib.name)
    if m:
        llvm_ver = int(m.group(1))
        clang = f"clang++-{llvm_ver}"
        import shutil
        if shutil.which(clang):
            return clang, llvm_ver

    return None, None


def find_best_ptxas_arch(gpu_cc):
    # Prefer the best architecture reported by nvcc, bounded by GPU CC.
    # If nvcc is unavailable, fall back to detected GPU CC.
    try:
        result = subprocess.run(
            ["nvcc", "--list-gpu-arch"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            max_arch = 0
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("compute_"):
                    arch_str = line.replace("compute_", "")
                    if arch_str.isdigit():
                        arch = int(arch_str)
                        if arch <= gpu_cc and arch > max_arch:
                            max_arch = arch
            if max_arch > 0:
                return max_arch
    except FileNotFoundError:
        logger.debug("nvcc not found in PATH")
    except subprocess.SubprocessError as e:
        logger.debug("nvcc --list-gpu-arch failed: %s", e)
    return gpu_cc


def cmd_diff(args):
    trace_a = Path(args.trace_a)
    trace_b = Path(args.trace_b)

    if not trace_a.exists():
        print(f"Error: Trace A not found: {trace_a}", file=sys.stderr)
        return 1
    if not trace_b.exists():
        print(f"Error: Trace B not found: {trace_b}", file=sys.stderr)
        return 1

    # Auto-detect session mode: if both paths are directories, infer --session
    session = getattr(args, "session", False)
    if trace_a.is_dir() and trace_b.is_dir():
        session = True

    differ = _find_lib.find_differ_binary()
    if not differ:
        print(
            "Error: prlx-diff binary not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    site_map = None
    if args.map:
        site_map = Path(args.map)
        if not site_map.exists():
            print(f"Warning: Site map not found: {site_map}", file=sys.stderr)
            site_map = None
    else:
        site_map = find_site_map(trace_a.parent)
        if site_map:
            print(f"Auto-detected site map: {site_map}")
            # Staleness check: warn if site map is older than both traces
            try:
                map_mtime = site_map.stat().st_mtime
                a_mtime = trace_a.stat().st_mtime
                b_mtime = trace_b.stat().st_mtime
                if map_mtime < min(a_mtime, b_mtime):
                    print(
                        "Warning: Site map is older than both traces, "
                        "it may be stale. Recompile or pass --map explicitly.",
                        file=sys.stderr,
                    )
            except OSError as e:
                logger.debug("Could not check site map staleness: %s", e)
        else:
            print("Note: No site map found. Divergences will show site IDs only.")
            print("      To see source locations, provide --map prlx-sites.json")

    cmd = [str(differ), str(trace_a), str(trace_b)]

    if site_map and site_map.exists():
        cmd.extend(["--map", str(site_map)])
    if args.values:
        cmd.append("--values")
    if args.verbose:
        cmd.append("--verbose")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.lookahead:
        cmd.extend(["--lookahead", str(args.lookahead)])
    if args.max_shown:
        cmd.extend(["-n", str(args.max_shown)])
    if hasattr(args, "tui") and args.tui:
        cmd.append("--tui")
    if hasattr(args, "float") and getattr(args, "float", False):
        cmd.append("--float")
    if hasattr(args, "force") and args.force:
        cmd.append("--force")
    if session:
        cmd.append("--session")

    return subprocess.call(cmd)


def cmd_run(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    if args.output:
        trace_file = args.output
    else:
        trace_file = f"{binary.stem}.prlx"

    env = os.environ.copy()
    env["PRLX_TRACE"] = trace_file

    print(f"Running {binary} with tracing enabled...")
    print(f"Trace output: {trace_file}")

    cmd = [str(binary)] + args.binary_args
    result = subprocess.call(cmd, env=env)

    if result == 0 and Path(trace_file).exists():
        print(f"Trace generated: {trace_file}")

    return result


def cmd_check(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a = Path(tmpdir) / "trace_a.prlx"
        trace_b = Path(tmpdir) / "trace_b.prlx"

        print("Non-determinism check: running the same binary twice and diffing traces.")
        print("To compare different inputs, use: prlx run + prlx run + prlx diff\n")
        print("=== Run A ===")
        env = os.environ.copy()
        env["PRLX_TRACE"] = str(trace_a)
        cmd = [str(binary)] + args.binary_args
        result_a = subprocess.call(cmd, env=env)

        if result_a != 0:
            print(f"Error: Run A failed with exit code {result_a}",
                  file=sys.stderr)
            return result_a
        if not trace_a.exists():
            print("Error: Run A did not generate a trace file",
                  file=sys.stderr)
            return 1

        print("\n=== Run B ===")
        env["PRLX_TRACE"] = str(trace_b)
        result_b = subprocess.call(cmd, env=env)

        if result_b != 0:
            print(f"Error: Run B failed with exit code {result_b}",
                  file=sys.stderr)
            return result_b
        if not trace_b.exists():
            print("Error: Run B did not generate a trace file",
                  file=sys.stderr)
            return 1

        print("\n=== Differential Analysis ===")

        class DiffArgs:
            def __init__(self):
                self.trace_a = trace_a
                self.trace_b = trace_b
                self.map = args.map
                self.values = args.values
                self.verbose = args.verbose
                self.limit = args.limit
                self.lookahead = args.lookahead
                self.max_shown = args.max_shown
                self.tui = getattr(args, "tui", False)
                self.float = getattr(args, "float", False)
                self.force = getattr(args, "force", False)
                self.session = False

        return cmd_diff(DiffArgs())


def cmd_compile(args):
    source = Path(args.source)

    if not source.exists():
        print(f"Error: Source file not found: {source}", file=sys.stderr)
        return 1

    pass_lib = _find_lib.find_pass_plugin()
    if not pass_lib:
        print(
            "Error: LLVM pass not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    clang, llvm_ver = find_clang_for_pass(pass_lib)
    if not clang:
        print("Error: No matching clang found for LLVM pass.", file=sys.stderr)
        print("       Install clang matching your LLVM version "
              "(e.g., apt install clang-20).", file=sys.stderr)
        return 1

    gpu_cc = detect_gpu_arch()
    ptxas_arch = find_best_ptxas_arch(gpu_cc) if gpu_cc else 90

    if args.arch:
        ptxas_arch = args.arch

    sm_str = f"sm_{ptxas_arch}"

    if args.output:
        output = args.output
    else:
        output = source.stem

    include_dirs = _find_lib.find_include_dirs()
    if not include_dirs:
        print("Warning: Could not find prlx headers. "
              "Compilation may fail.", file=sys.stderr)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cu", delete=False, prefix="prlx_unified_"
    ) as f:
        unified_path = f.name
        f.write("// Auto-generated unified compilation wrapper\n")
        f.write('#include "prlx_runtime.cu"\n')
        f.write('#include "prlx_host.cu"\n')
        f.write(f'#include "{source.resolve()}"\n')

    try:
        cmd = [
            clang,
            f"-fpass-plugin={pass_lib}",
            f"--cuda-gpu-arch={sm_str}",
            f"--cuda-include-ptx={sm_str}",
            "-g", "-O0",
        ]

        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])
        cmd.extend(["-I", str(source.resolve().parent)])

        for inc in (args.include or []):
            cmd.extend(["-I", inc])

        cmd.extend([unified_path, "-lcudart", "-lm", "-o", str(output)])

        if args.extra:
            cmd.extend(args.extra)

        if args.verbose:
            print(f"GPU: cc{gpu_cc or '?'}, compiling for "
                  f"{sm_str} + PTX (JIT forward-compat)")
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not args.verbose)

        if result.returncode != 0:
            if not args.verbose and result.stderr:
                sys.stderr.buffer.write(result.stderr)
            return result.returncode

        print(f"Compiled: {output} (instrumented, {sm_str}+PTX)")
        return 0

    finally:
        os.unlink(unified_path)


def cmd_session(args):
    subcmd = args.session_command
    if not subcmd:
        print("Usage: prlx session {capture,inspect,diff} ...", file=sys.stderr)
        return 1

    if subcmd == "capture":
        return cmd_session_capture(args)
    elif subcmd == "inspect":
        return cmd_session_inspect(args)
    elif subcmd == "diff":
        return cmd_session_diff(args)

    print(f"Unknown session subcommand: {subcmd}", file=sys.stderr)
    return 1


def cmd_session_capture(args):
    binary = Path(args.binary)

    if not binary.exists():
        print(f"Error: Binary not found: {binary}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else Path(f"{binary.stem}_session")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PRLX_SESSION"] = str(output_dir.resolve())

    cmd = [str(binary)] + (args.binary_args or [])
    print(f"Running {binary} with session capture to {output_dir}...")
    result = subprocess.call(cmd, env=env)

    manifest = output_dir / "session.json"
    if result == 0 and manifest.exists():
        print(f"Session captured: {output_dir}")
        with open(manifest) as f:
            launches = json.load(f)
        print(f"  {len(launches)} kernel launch(es) recorded")
    elif result == 0:
        print(f"Warning: Binary exited successfully but no session.json found in {output_dir}",
              file=sys.stderr)

    return result


def cmd_session_inspect(args):
    session_dir = Path(args.session_dir)
    manifest_path = session_dir / "session.json"

    if not manifest_path.exists():
        print(f"Error: No session.json in {session_dir}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        launches = json.load(f)

    print(f"Session: {session_dir}")
    print(f"Launches: {len(launches)}")
    print()

    # Table header
    print(f"{'#':<4} {'Kernel':<40} {'Grid':<16} {'Block':<16} {'File'}")
    print("-" * 100)

    for i, launch in enumerate(launches):
        grid = "x".join(str(g) for g in launch.get("grid", []))
        block = "x".join(str(b) for b in launch.get("block", []))
        kernel = launch.get("kernel", "?")
        fname = launch.get("file", "?")
        launch_idx = launch.get("launch", i)
        print(f"{launch_idx:<4} {kernel:<40} {grid:<16} {block:<16} {fname}")

    return 0


def cmd_session_diff(args):
    """Delegate to cmd_diff with session=True."""

    class DiffArgs:
        def __init__(self):
            self.trace_a = args.session_a
            self.trace_b = args.session_b
            self.map = getattr(args, "map", None)
            self.values = getattr(args, "values", False)
            self.verbose = getattr(args, "verbose", False)
            self.limit = getattr(args, "limit", None)
            self.lookahead = getattr(args, "lookahead", None)
            self.max_shown = getattr(args, "max_shown", 10)
            self.tui = getattr(args, "tui", False)
            self.float = getattr(args, "float", False)
            self.force = getattr(args, "force", False)
            self.session = True

    return cmd_diff(DiffArgs())


def cmd_triton(args):
    if args.info or args.check_env:
        print("=== PRLX Triton Integration ===")

        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            print(f"  LLVM Pass Plugin: {pass_plugin}")
        else:
            print("  LLVM Pass Plugin: NOT FOUND")

        runtime_lib = _find_lib.find_runtime_library()
        if runtime_lib:
            print(f"  Runtime Library:  {runtime_lib}")
        else:
            print("  Runtime Library:  NOT FOUND")

        bitcode = _find_lib.find_runtime_bitcode()
        if bitcode:
            print(f"  Runtime Bitcode:  {bitcode}")
        else:
            print("  Runtime Bitcode:  NOT FOUND")

        differ = _find_lib.find_differ_binary()
        if differ:
            print(f"  Differ Binary:    {differ}")
        else:
            print("  Differ Binary:    NOT FOUND")

        opt = _find_lib.find_opt_binary()
        if opt:
            print(f"  opt:              {opt}")
        else:
            print("  opt:              NOT FOUND (apt install llvm-20)")

        llvm_link = _find_lib.find_llvm_link_binary()
        if llvm_link:
            print(f"  llvm-link:        {llvm_link}")
        else:
            print("  llvm-link:        NOT FOUND (apt install llvm-20)")

        try:
            import triton
            print(f"  Triton:           {triton.__version__}")
        except ImportError:
            print("  Triton:           NOT INSTALLED (pip install triton)")

        try:
            import prlx
            print(f"  prlx package:     {prlx.__version__}")
        except ImportError:
            print("  prlx package:     NOT INSTALLED")

        print()
        print("Quick start:")
        print("  python -c 'import prlx; prlx.enable()'")
        return 0

    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}", file=sys.stderr)
            return 1

        env = os.environ.copy()

        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            existing = env.get("LLVM_PASS_PLUGINS", "")
            env["LLVM_PASS_PLUGINS"] = (
                f"{existing};{pass_plugin}" if existing else str(pass_plugin)
            )
        else:
            print("Warning: LLVM pass plugin not found", file=sys.stderr)

        cmd = [sys.executable, str(script_path)] + args.script_args
        return subprocess.call(cmd, env=env)

    print("Usage:")
    print("  prlx triton --info           Show integration status")
    print("  prlx triton script.py        Run script with instrumentation")
    return 0


def cmd_assert(args):
    """CI regression gate: compare two traces and exit 0/1 based on divergence threshold."""

    # Resolve trace paths: golden mode vs positional mode
    if args.golden:
        trace_a = Path(args.golden)
        if not args.trace:
            print("Error: --golden requires a test trace as positional argument",
                  file=sys.stderr)
            return 1
        trace_b = Path(args.trace)
    else:
        if not args.trace_a_pos or not args.trace:
            print("Error: assert requires two trace files (or --golden <golden> <test>)",
                  file=sys.stderr)
            return 1
        trace_a = Path(args.trace_a_pos)
        trace_b = Path(args.trace)

    if not trace_a.exists():
        print(f"Error: Trace A not found: {trace_a}", file=sys.stderr)
        return 1
    if not trace_b.exists():
        print(f"Error: Trace B not found: {trace_b}", file=sys.stderr)
        return 1

    differ = _find_lib.find_differ_binary()
    if not differ:
        print(
            "Error: prlx-diff binary not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    threshold = args.max_divergences

    # Build prlx-diff command, always request JSON for structured output
    cmd = [str(differ), str(trace_a), str(trace_b), "--json"]

    if threshold > 0:
        cmd.extend(["--max-allowed-divergences", str(threshold)])

    if args.ignore_active_mask:
        cmd.append("--ignore-active-mask")

    if args.map:
        cmd.extend(["--map", str(args.map)])

    if args.session:
        cmd.append("--session")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if args.json:
        # Raw JSON passthrough to stdout
        sys.stdout.write(result.stdout)
        return result.returncode

    # Parse JSON and print human-readable summary.
    # The Rust binary may print non-JSON lines (e.g. "Loading traces...")
    # before the JSON object, so scan for the first decodable JSON object.
    stdout = result.stdout
    try:
        data = None
        decoder = json.JSONDecoder()

        for i, ch in enumerate(stdout):
            if ch != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stdout[i:])
                if isinstance(candidate, dict):
                    data = candidate
                    break
            except json.JSONDecodeError:
                continue

        if data is None:
            raise json.JSONDecodeError("no JSON object found", stdout, 0)

        count = data.get("total_divergences", data.get("divergence_count", 0))
    except (json.JSONDecodeError, KeyError):
        # If JSON parsing fails, fall back to exit code
        if result.returncode == 0:
            count = 0
        else:
            print("PRLX ASSERT: FAIL (could not parse prlx-diff output)",
                  file=sys.stderr)
            if result.stderr:
                sys.stderr.write(result.stderr)
            return result.returncode

    if count <= threshold:
        print(f"PRLX ASSERT: PASS ({count} divergences, threshold: {threshold})")
    else:
        print(f"PRLX ASSERT: FAIL ({count} divergences, threshold: {threshold})")

    return result.returncode


def cmd_flamegraph(args):
    """Export divergences to Chrome Trace Format for flamegraph visualization."""

    trace_a = Path(args.trace_a)
    trace_b = Path(args.trace_b)

    if not trace_a.exists():
        print(f"Error: Trace A not found: {trace_a}", file=sys.stderr)
        return 1
    if not trace_b.exists():
        print(f"Error: Trace B not found: {trace_b}", file=sys.stderr)
        return 1

    differ = _find_lib.find_differ_binary()
    if not differ:
        print(
            "Error: prlx-diff binary not found.\n"
            "       Install with: pip install prlx",
            file=sys.stderr,
        )
        return 1

    output = args.output or "prlx-flamegraph.json"

    cmd = [
        str(differ),
        "--export-flamegraph", str(output),
        str(trace_a), str(trace_b),
    ]

    if args.map:
        cmd.extend(["--map", str(args.map)])
    if args.values:
        cmd.append("--values")
    if args.force:
        cmd.append("--force")
    if args.session:
        cmd.append("--session")

    result = subprocess.call(cmd)

    if result == 0:
        output_path = Path(output)
        if output_path.exists():
            size = output_path.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            print(f"Flamegraph written: {output} ({size_str})")
            print("Open in chrome://tracing or https://ui.perfetto.dev")

    return result


def cmd_pytorch(args):
    """PyTorch integration: info, run with instrumentation, or NVBit mode."""

    subcmd = getattr(args, "pytorch_command", None)

    if subcmd == "run":
        return cmd_pytorch_run(args)
    elif subcmd == "watch":
        return cmd_pytorch_watch(args)
    elif subcmd == "diff-steps":
        return cmd_pytorch_diff_steps(args)
    elif subcmd == "report":
        return cmd_pytorch_report(args)

    if args.info:
        print("=== PRLX PyTorch Integration ===")

        # Check torch availability
        try:
            import torch
            print(f"  PyTorch:          {torch.__version__}")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"  CUDA Device:      {device_name}")
            else:
                print("  CUDA Device:      NOT AVAILABLE")
        except ImportError:
            print("  PyTorch:          NOT INSTALLED (pip install torch)")

        # Check pass plugin
        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            print(f"  LLVM Pass Plugin: {pass_plugin}")
        else:
            print("  LLVM Pass Plugin: NOT FOUND")

        # Check NVBit tool
        nvbit_lib = _find_nvbit_tool()
        if nvbit_lib:
            print(f"  NVBit Tool:       {nvbit_lib}")
        else:
            print("  NVBit Tool:       NOT FOUND")

        # Check differ
        differ = _find_lib.find_differ_binary()
        if differ:
            print(f"  Differ Binary:    {differ}")
        else:
            print("  Differ Binary:    NOT FOUND")

        print()
        print("Quick start:")
        print("  prlx pytorch run script.py           Run with LLVM pass instrumentation")
        print("  prlx pytorch run --nvbit script.py   Run with NVBit LD_PRELOAD")
        print("  prlx pytorch watch script.py         Watch training for anomalies")
        return 0

    print("Usage:")
    print("  prlx pytorch --info              Show integration status")
    print("  prlx pytorch run script.py       Run with instrumentation")
    print("  prlx pytorch run --nvbit script.py  Run with NVBit LD_PRELOAD")
    print("  prlx pytorch watch script.py     Watch training for anomalies")
    print("  prlx pytorch diff-steps DIR --step-a N --step-b M")
    print("  prlx pytorch report DIR [--json]")
    return 0


def cmd_pytorch_run(args):
    """Run a Python script with PyTorch instrumentation."""

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()

    if args.nvbit:
        # NVBit mode: LD_PRELOAD the NVBit tool
        nvbit_lib = _find_nvbit_tool()
        if not nvbit_lib:
            print(
                "Error: NVBit tool (libprlx_nvbit.so) not found.\n"
                "       Build with: cmake --build build --target prlx_nvbit",
                file=sys.stderr,
            )
            return 1
        existing_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (
            f"{existing_preload}:{nvbit_lib}" if existing_preload
            else str(nvbit_lib)
        )
        print(f"Running with NVBit instrumentation: {script_path}")
    else:
        # LLVM pass mode
        pass_plugin = _find_lib.find_pass_plugin()
        if pass_plugin:
            existing = env.get("LLVM_PASS_PLUGINS", "")
            env["LLVM_PASS_PLUGINS"] = (
                f"{existing};{pass_plugin}" if existing else str(pass_plugin)
            )
        else:
            print("Warning: LLVM pass plugin not found", file=sys.stderr)
        print(f"Running with LLVM pass instrumentation: {script_path}")

    if args.output:
        env["PRLX_TRACE"] = args.output

    cmd = [sys.executable, str(script_path)] + (args.script_args or [])
    return subprocess.call(cmd, env=env)


def cmd_pytorch_watch(args):
    """Run a training script with anomaly monitoring enabled."""

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PRLX_PYTORCH_WATCH"] = "1"

    if args.output_dir:
        env["PRLX_TRACE_DIR"] = args.output_dir
    if args.threshold:
        env["PRLX_LOSS_THRESHOLD"] = str(args.threshold)
    if args.grad_threshold:
        env["PRLX_GRAD_NORM_THRESHOLD"] = str(args.grad_threshold)
    if args.window:
        env["PRLX_WINDOW_SIZE"] = str(args.window)

    print(f"Watching training script: {script_path}")
    print(f"  Output dir: {env.get('PRLX_TRACE_DIR', './prlx_traces')}")
    print(f"  Loss threshold: {env.get('PRLX_LOSS_THRESHOLD', '2.0')}x moving average")
    print()

    cmd = [sys.executable, str(script_path)] + (args.script_args or [])
    return subprocess.call(cmd, env=env)


def cmd_pytorch_diff_steps(args):
    """Diff two training step captures using session mode."""

    trace_dir = Path(args.trace_dir)
    step_a_dir = trace_dir / f"step_{args.step_a}"
    step_b_dir = trace_dir / f"step_{args.step_b}"

    if not step_a_dir.exists():
        print(f"Error: Step directory not found: {step_a_dir}", file=sys.stderr)
        return 1
    if not step_b_dir.exists():
        print(f"Error: Step directory not found: {step_b_dir}", file=sys.stderr)
        return 1

    print(f"Diffing step {args.step_a} vs step {args.step_b}")
    print(f"  Step A: {step_a_dir}")
    print(f"  Step B: {step_b_dir}")
    print()

    class DiffArgs:
        def __init__(self):
            self.trace_a = str(step_a_dir)
            self.trace_b = str(step_b_dir)
            self.map = getattr(args, "map", None)
            self.values = getattr(args, "values", False)
            self.verbose = getattr(args, "verbose", False)
            self.limit = getattr(args, "limit", None)
            self.lookahead = getattr(args, "lookahead", None)
            self.max_shown = getattr(args, "max_shown", 10)
            self.tui = False
            self.float = False
            self.force = False
            self.session = True

    return cmd_diff(DiffArgs())


def cmd_pytorch_report(args):
    """Parse anomaly log and summarize training monitoring results."""

    trace_dir = Path(args.trace_dir)

    # Check for report file
    report_path = trace_dir / "training_report.json"
    anomaly_log_path = trace_dir / "anomaly_log.jsonl"

    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}", file=sys.stderr)
        return 1

    # Try training_report.json first
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        print("=== PRLX Training Report ===")
        print(f"  Output dir:       {report.get('output_dir', str(trace_dir))}")
        print(f"  Total anomalies:  {report.get('total_anomalies', 0)}")
        print(f"  Captures:         {report.get('captures', 0)}")
        print(f"  Loss history:     {report.get('loss_history_length', 0)} steps")
        print()

        anomalies = report.get("anomalies", [])
        if anomalies:
            print("Anomalies:")
            for a in anomalies:
                print(f"  Step {a['step']}: {a['type']} "
                      f"(value={a['value']:.6g}, threshold={a['threshold']:.6g}, "
                      f"moving_avg={a['moving_average']:.6g})")
        else:
            print("No anomalies detected.")

        return 0

    # Fall back to anomaly_log.jsonl
    if anomaly_log_path.exists():
        anomalies = []
        with open(anomaly_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    anomalies.append(json.loads(line))

        if args.json:
            print(json.dumps({"anomalies": anomalies}, indent=2))
            return 0

        print("=== PRLX Anomaly Log ===")
        print(f"  Total anomalies: {len(anomalies)}")
        print()

        for a in anomalies:
            print(f"  Step {a['step']}: {a['type']} "
                  f"(value={a['value']:.6g}, threshold={a['threshold']:.6g}, "
                  f"moving_avg={a['moving_average']:.6g})")

        return 0

    print(f"Error: No training report or anomaly log found in {trace_dir}",
          file=sys.stderr)
    return 1


def _find_nvbit_tool():
    """Locate libprlx_nvbit.so for NVBit LD_PRELOAD mode."""
    home = os.environ.get("PRLX_HOME")
    if home:
        p = Path(home) / "lib" / "libprlx_nvbit.so"
        if p.exists():
            return p.resolve()

    root = _find_lib._project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "nvbit_tool" / "libprlx_nvbit.so"
            if p.exists():
                return p.resolve()

    import shutil
    path = shutil.which("libprlx_nvbit.so")
    if path:
        return Path(path)

    return None


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="PRLX - Differential debugger for CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two traces
  prlx diff trace_a.prlx trace_b.prlx

  # Compare with source location mapping
  prlx diff trace_a.prlx trace_b.prlx --map prlx-sites.json

  # Compare session directories (multi-launch traces)
  prlx diff session_a/ session_b/ --session

  # Run a binary with tracing
  prlx run ./my_kernel --output my_trace.prlx

  # Run twice and diff (detect non-determinism)
  prlx check ./my_kernel

  # Compile with automatic instrumentation
  prlx compile kernel.cu -o my_kernel

  # Multi-kernel session operations
  prlx session capture ./my_pipeline -o /tmp/sess_a
  prlx session inspect /tmp/sess_a
  prlx session diff /tmp/sess_a /tmp/sess_b

  # Triton integration info
  prlx triton --info

  # CI regression gate
  prlx assert trace_a.prlx trace_b.prlx
  prlx assert --golden golden.prlx test.prlx --max-divergences 5
  prlx assert trace_a.prlx trace_b.prlx --json

  # Export flamegraph for Chrome tracing
  prlx flamegraph trace_a.prlx trace_b.prlx -o divergences.json

  # PyTorch integration
  prlx pytorch --info
  prlx pytorch run train.py
  prlx pytorch run --nvbit train.py

  # Training loop monitoring
  prlx pytorch watch train.py --threshold 3.0 --output-dir ./traces
  prlx pytorch diff-steps ./traces --step-a 100 --step-b 105
  prlx pytorch report ./traces --json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # diff
    p_diff = subparsers.add_parser("diff", help="Compare two trace files")
    p_diff.add_argument("trace_a", help="First trace file (baseline)")
    p_diff.add_argument("trace_b", help="Second trace file (compare)")
    p_diff.add_argument("--map", help="Site mapping file (prlx-sites.json)")
    p_diff.add_argument("-n", "--max-shown", type=int, default=10,
                        help="Max divergences to display")
    p_diff.add_argument("--values", action="store_true",
                        help="Compare operand values")
    p_diff.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    p_diff.add_argument("-l", "--limit", type=int,
                        help="Max divergences to collect")
    p_diff.add_argument("--lookahead", type=int,
                        help="Lookahead window size")
    p_diff.add_argument("--tui", action="store_true",
                        help="Launch interactive TUI viewer")
    p_diff.add_argument("--float", action="store_true",
                        help="Display snapshot operands as floats")
    p_diff.add_argument("--force", action="store_true",
                        help="Skip kernel name check (compare different variants)")
    p_diff.add_argument("--session", action="store_true",
                        help="Compare session directories (multi-launch traces)")

    # run
    p_run = subparsers.add_parser("run",
                                  help="Run a binary with tracing enabled")
    p_run.add_argument("binary", help="Binary to execute")
    p_run.add_argument("-o", "--output", help="Output trace file")
    p_run.add_argument("binary_args", nargs="*",
                       help="Arguments to pass to binary")

    # check
    p_check = subparsers.add_parser(
        "check",
        help="Run binary twice with identical inputs and diff (tests non-determinism)",
    )
    p_check.add_argument("binary", help="Binary to execute")
    p_check.add_argument("--map",
                         help="Site mapping file (prlx-sites.json)")
    p_check.add_argument("-n", "--max-shown", type=int, default=10,
                         help="Max divergences to display")
    p_check.add_argument("--values", action="store_true",
                         help="Compare operand values")
    p_check.add_argument("-v", "--verbose", action="store_true",
                         help="Verbose output")
    p_check.add_argument("-l", "--limit", type=int,
                         help="Max divergences to collect")
    p_check.add_argument("--lookahead", type=int,
                         help="Lookahead window size")
    p_check.add_argument("--tui", action="store_true",
                         help="Launch interactive TUI viewer")
    p_check.add_argument("--float", action="store_true",
                         help="Display snapshot operands as floats")
    p_check.add_argument("binary_args", nargs="*",
                         help="Arguments to pass to binary")

    # compile
    p_compile = subparsers.add_parser(
        "compile", help="Compile CUDA source with instrumentation")
    p_compile.add_argument("source", help="CUDA source file (.cu)")
    p_compile.add_argument("-o", "--output", help="Output binary name")
    p_compile.add_argument("-I", "--include", action="append",
                           help="Include directories")
    p_compile.add_argument("--arch", type=int,
                           help="CUDA SM architecture (e.g. 90)")
    p_compile.add_argument("-v", "--verbose", action="store_true",
                           help="Show compilation command")
    p_compile.add_argument("--extra", nargs="*",
                           help="Extra compiler flags")

    # session
    p_session = subparsers.add_parser(
        "session", help="Multi-kernel session operations")
    session_subs = p_session.add_subparsers(dest="session_command",
                                            help="Session subcommand")

    # session capture
    p_sess_capture = session_subs.add_parser(
        "capture", help="Capture a session (run binary with PRLX_SESSION)")
    p_sess_capture.add_argument("binary", help="Binary to execute")
    p_sess_capture.add_argument("-o", "--output",
                                help="Output session directory")
    p_sess_capture.add_argument("binary_args", nargs="*",
                                help="Arguments to pass to binary")

    # session inspect
    p_sess_inspect = session_subs.add_parser(
        "inspect", help="Inspect a session manifest")
    p_sess_inspect.add_argument("session_dir",
                                help="Session directory to inspect")

    # session diff
    p_sess_diff = session_subs.add_parser(
        "diff", help="Diff two session directories")
    p_sess_diff.add_argument("session_a", help="First session directory")
    p_sess_diff.add_argument("session_b", help="Second session directory")
    p_sess_diff.add_argument("--map", help="Site mapping file")
    p_sess_diff.add_argument("-n", "--max-shown", type=int, default=10,
                             help="Max divergences to display")
    p_sess_diff.add_argument("--values", action="store_true",
                             help="Compare operand values")
    p_sess_diff.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output")
    p_sess_diff.add_argument("-l", "--limit", type=int,
                             help="Max divergences to collect")
    p_sess_diff.add_argument("--lookahead", type=int,
                             help="Lookahead window size")
    p_sess_diff.add_argument("--float", action="store_true",
                             help="Display snapshot operands as floats")
    p_sess_diff.add_argument("--force", action="store_true",
                             help="Skip kernel name check")

    # triton
    p_triton = subparsers.add_parser("triton",
                                     help="Triton/Python integration")
    p_triton.add_argument("--info", action="store_true",
                          help="Show integration status")
    p_triton.add_argument("--check-env", action="store_true",
                          help="Verify environment setup")
    p_triton.add_argument("script", nargs="?",
                          help="Python script to run with Triton integration")
    p_triton.add_argument("script_args", nargs="*",
                          help="Arguments to pass to the script")

    # assert
    p_assert = subparsers.add_parser(
        "assert", help="CI regression gate: compare traces against threshold")
    p_assert.add_argument("trace_a_pos", nargs="?",
                          help="First trace file (baseline); "
                               "omit when using --golden")
    p_assert.add_argument("trace", nargs="?",
                          help="Second trace file (test)")
    p_assert.add_argument("--golden",
                          help="Golden reference trace (trace_a); "
                               "positional arg becomes trace_b")
    p_assert.add_argument("--max-divergences", type=int, default=0,
                          help="Maximum allowed divergences (default: 0 = identical)")
    p_assert.add_argument("--json", action="store_true",
                          help="Raw JSON passthrough to stdout")
    p_assert.add_argument("--ignore-active-mask", action="store_true",
                          help="Ignore active mask divergences")
    p_assert.add_argument("--session", action="store_true",
                          help="Compare session directories")
    p_assert.add_argument("--map",
                          help="Site mapping file (prlx-sites.json)")

    # flamegraph
    p_flamegraph = subparsers.add_parser(
        "flamegraph",
        help="Export divergences to Chrome Trace Format")
    p_flamegraph.add_argument("trace_a", help="First trace file (baseline)")
    p_flamegraph.add_argument("trace_b", help="Second trace file (compare)")
    p_flamegraph.add_argument("-o", "--output",
                              help="Output JSON file "
                                   "(default: prlx-flamegraph.json)")
    p_flamegraph.add_argument("--map",
                              help="Site mapping file (prlx-sites.json)")
    p_flamegraph.add_argument("--values", action="store_true",
                              help="Include operand values")
    p_flamegraph.add_argument("--force", action="store_true",
                              help="Skip kernel name check")
    p_flamegraph.add_argument("--session", action="store_true",
                              help="Compare session directories")

    # pytorch
    p_pytorch = subparsers.add_parser(
        "pytorch", help="PyTorch integration and instrumentation")
    p_pytorch.add_argument("--info", action="store_true",
                           help="Show PyTorch integration status")

    pytorch_subs = p_pytorch.add_subparsers(dest="pytorch_command",
                                             help="PyTorch subcommand")

    # pytorch run (script execution)
    p_pt_run = pytorch_subs.add_parser(
        "run", help="Run a Python script with PyTorch instrumentation")
    p_pt_run.add_argument("script", help="Python script to run")
    p_pt_run.add_argument("--nvbit", action="store_true",
                          help="Use NVBit LD_PRELOAD instead of LLVM pass")
    p_pt_run.add_argument("-o", "--output",
                          help="Output trace file (sets PRLX_TRACE)")
    p_pt_run.add_argument("script_args", nargs="*",
                          help="Arguments to pass to the script")

    # pytorch watch
    p_pt_watch = pytorch_subs.add_parser(
        "watch", help="Run training script with anomaly monitoring")
    p_pt_watch.add_argument("script", help="Training script to run")
    p_pt_watch.add_argument("--output-dir",
                            help="Directory for trace captures")
    p_pt_watch.add_argument("--threshold", type=float,
                            help="Loss spike threshold multiplier (default: 2.0)")
    p_pt_watch.add_argument("--grad-threshold", type=float,
                            help="Gradient norm threshold (default: 100.0)")
    p_pt_watch.add_argument("--window", type=int,
                            help="Moving average window size (default: 50)")
    p_pt_watch.add_argument("script_args", nargs="*",
                            help="Arguments to pass to the script")

    # pytorch diff-steps
    p_pt_diff = pytorch_subs.add_parser(
        "diff-steps", help="Diff two captured training steps")
    p_pt_diff.add_argument("trace_dir",
                           help="Trace output directory from watch")
    p_pt_diff.add_argument("--step-a", type=int, required=True,
                           help="First step number")
    p_pt_diff.add_argument("--step-b", type=int, required=True,
                           help="Second step number")
    p_pt_diff.add_argument("--map", help="Site mapping file")
    p_pt_diff.add_argument("--values", action="store_true",
                           help="Compare operand values")
    p_pt_diff.add_argument("-v", "--verbose", action="store_true",
                           help="Verbose output")

    # pytorch report
    p_pt_report = pytorch_subs.add_parser(
        "report", help="Show training monitoring report")
    p_pt_report.add_argument("trace_dir",
                             help="Trace output directory")
    p_pt_report.add_argument("--json", action="store_true",
                             help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "diff": cmd_diff,
        "compile": cmd_compile,
        "run": cmd_run,
        "check": cmd_check,
        "session": cmd_session,
        "triton": cmd_triton,
        "assert": cmd_assert,
        "flamegraph": cmd_flamegraph,
        "pytorch": cmd_pytorch,
    }
    handler = dispatch.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1
