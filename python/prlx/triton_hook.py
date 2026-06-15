"""
Triton compiler integration for prlx.

Intercepts LLVM IR between Triton's make_llir() and make_ptx() stages,
links device-side runtime bitcode, and runs the instrumentation pass.

Requires Triton >= 3.0 (knobs.runtime stages API).
"""

import os
import sys
import subprocess
import tempfile
import functools
from pathlib import Path
from typing import Optional

from ._log import get_logger

logger = get_logger(__name__)

from ._find_lib import (
    find_pass_plugin,
    find_runtime_bitcode,
    find_runtime_library,
    find_opt_binary,
    find_llvm_link_binary,
)


_installed = False
_original_launch = None
_pass_plugin: Optional[Path] = None
_runtime_bc: Optional[Path] = None
_opt_bin: Optional[Path] = None
_llvm_link_bin: Optional[Path] = None


def install(pass_plugin: Optional[str] = None, verbose: bool = True):
    global _installed, _pass_plugin, _runtime_bc, _opt_bin, _llvm_link_bin
    if _installed:
        return

    _pass_plugin = Path(pass_plugin) if pass_plugin else find_pass_plugin()
    _runtime_bc = find_runtime_bitcode()
    _opt_bin = find_opt_binary()
    _llvm_link_bin = find_llvm_link_binary()

    missing = []
    if not _pass_plugin or not _pass_plugin.exists():
        missing.append("libPrlxPass.so (build with: cmake --build build)")
    if not _runtime_bc or not _runtime_bc.exists():
        missing.append("prlx_runtime_nvptx.bc (build with: cmake --build build)")
    if not _opt_bin:
        missing.append("opt (install with: apt install llvm-20)")
    if not _llvm_link_bin:
        missing.append("llvm-link (install with: apt install llvm-20)")

    if missing:
        raise FileNotFoundError(
            "Missing components for Triton integration:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    logger.info("Pass plugin: %s", _pass_plugin)
    logger.info("Runtime BC:  %s", _runtime_bc)
    logger.info("opt:         %s", _opt_bin)
    logger.info("llvm-link:   %s", _llvm_link_bin)

    _hook_triton_stages(verbose)
    _wrap_triton_launch(verbose)

    _installed = True
    logger.info("Triton integration installed")


def _get_subprocess_timeout() -> int:
    """Get timeout for llvm-link/opt subprocesses (seconds)."""
    try:
        return int(os.environ.get("PRLX_OPT_TIMEOUT", "120"))
    except ValueError:
        return 120


def _instrument_llvm_ir(llvm_ir: str) -> str:
    """input.ll -> llvm-link (+ runtime BC) -> opt (+ pass) -> output.ll"""
    timeout = _get_subprocess_timeout()
    with tempfile.TemporaryDirectory(prefix="prlx_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.ll")
        linked_path = os.path.join(tmpdir, "linked.ll")
        output_path = os.path.join(tmpdir, "output.ll")

        with open(input_path, "w") as f:
            f.write(llvm_ir)

        # --suppress-warnings: Triton's extended data layout vs clang's default
        link_cmd = [
            str(_llvm_link_bin),
            input_path,
            str(_runtime_bc),
            "-S",
            "-o", linked_path,
            "--suppress-warnings",
        ]
        result = subprocess.run(
            link_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"llvm-link failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )

        opt_cmd = [
            str(_opt_bin),
            "-load-pass-plugin", str(_pass_plugin),
            "-passes=prlx",
            linked_path,
            "-S",
            "-o", output_path,
        ]
        result = subprocess.run(
            opt_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"prlx instrumentation pass failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )

        with open(output_path) as f:
            instrumented = f.read()

        for line in result.stderr.splitlines():
            if "[prlx]" in line:
                logger.info("%s", line.strip())

        return instrumented


def _hook_triton_stages(verbose: bool):
    from triton import knobs

    def prlx_stages_hook(backend, stages, options, language, capability):
        if "llir" not in stages:
            return

        original_llir = stages["llir"]

        def instrumented_llir(src, metadata):
            llvm_ir = original_llir(src, metadata)
            if os.environ.get("PRLX_ENABLED", "1") == "0":
                return llvm_ir
            return _instrument_llvm_ir(llvm_ir)

        stages["llir"] = instrumented_llir

    knobs.runtime.add_stages_inspection_hook = prlx_stages_hook

    logger.info("Hooked Triton stages API")


def _wrap_triton_launch(verbose: bool):
    global _original_launch

    from triton import knobs
    from triton.runtime import JITFunction

    if _original_launch is not None:
        return

    rt_lib = find_runtime_library()
    if rt_lib is None:
        logger.info("Runtime library not found, manual buffer management required")
        return

    from .runtime import PrlxRuntime
    from ._module_binder import ModuleBinder

    runtime = PrlxRuntime(str(rt_lib))
    binder = ModuleBinder(runtime, verbose=verbose)

    # Install launch_enter_hook: fires RIGHT BEFORE cuLaunchKernel.
    # This is where we bind device globals (g_prlx_buffer, etc.) into
    # the Triton kernel's CUmodule via CUDA Driver API.
    def _prlx_launch_enter(metadata):
        if os.environ.get("PRLX_ENABLED", "1") == "0":
            return
        fn_handle = metadata.data.get("function")
        if fn_handle is not None:
            binder.bind_module(int(fn_handle))

    knobs.runtime.launch_enter_hook = _prlx_launch_enter

    logger.info("launch_enter_hook installed")

    _original_launch = JITFunction.run

    @functools.wraps(_original_launch)
    def instrumented_run(self, *args, **kwargs):
        if os.environ.get("PRLX_ENABLED", "1") == "0":
            return _original_launch(self, *args, **kwargs)

        grid = kwargs.get("grid", (1, 1, 1))
        if callable(grid):
            # Triton typically passes grid as a lambda; evaluate it to get
            # actual dimensions. Pass kwargs as the meta dict (Triton convention).
            try:
                resolved = grid(kwargs)
                if isinstance(resolved, (tuple, list)):
                    grid_dim = tuple(resolved) + (1,) * (3 - len(resolved))
                else:
                    grid_dim = (int(resolved), 1, 1)
            except Exception as exc:
                # If grid lambda fails (e.g. needs meta keys we don't have),
                # fall back to num_warps-based estimate to avoid OOB writes.
                # Set PRLX_STRICT_GRID=1 to fail fast instead.
                if os.environ.get("PRLX_STRICT_GRID", "0") == "1":
                    raise RuntimeError(
                        f"Failed to resolve Triton grid for kernel "
                        f"{getattr(self, '__name__', 'triton_kernel')}: {exc}"
                    ) from exc
                num_warps = int(kwargs.get("num_warps", 4))
                logger.warning(
                    "Failed to resolve Triton grid lambda; "
                    "using fallback grid=(%d, 1, 1): %s",
                    num_warps, exc,
                )
                grid_dim = (num_warps, 1, 1)
        elif isinstance(grid, (tuple, list)):
            grid_dim = tuple(grid) + (1,) * (3 - len(grid))
        else:
            grid_dim = (int(grid), 1, 1)

        block_dim = (kwargs.get("num_warps", 4) * 32, 1, 1)
        kernel_name = getattr(self, "__name__", "triton_kernel")

        runtime.pre_launch(kernel_name, grid_dim, block_dim)
        # Invalidate binder cache, pre_launch allocated new buffers
        binder.invalidate()
        result = _original_launch(self, *args, **kwargs)
        runtime.post_launch()

        return result

    JITFunction.run = instrumented_run
    logger.info("Kernel launch wrapper installed")


def uninstall():
    """Remove all Triton integration hooks."""
    global _installed, _original_launch

    if _original_launch is not None:
        from triton.runtime import JITFunction
        JITFunction.run = _original_launch
        _original_launch = None

    from triton import knobs
    knobs.runtime.add_stages_inspection_hook = None
    knobs.runtime.launch_enter_hook = None

    _installed = False
