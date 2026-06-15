"""
PyTorch integration for prlx.

Three-tier instrumentation strategy:
  Tier 1, Triton via torch.compile: Delegate to triton_hook.install().
  Tier 2, load_inline hook: Monkey-patch torch.utils.cpp_extension.load_inline
           to inject -fpass-plugin=libPrlxPass.so into extra_cuda_cflags.
  Tier 3, NVBit fallback: Set LD_PRELOAD=libprlx_nvbit.so for SASS-level
           binary instrumentation.

Requires PyTorch >= 2.0.
"""

import os
import sys
import functools
from pathlib import Path
from typing import Optional

from ._log import get_logger

logger = get_logger(__name__)

from ._find_lib import (
    find_pass_plugin,
    find_runtime_library,
    find_nvbit_library,
)

_installed = False
_original_load_inline = None
_tier1_active = False
_tier2_active = False
_tier3_active = False


def install(
    instrument_triton: bool = True,
    instrument_extensions: bool = True,
    nvbit_precompiled: bool = False,
    verbose: bool = True,
):
    """Hook into PyTorch for GPU kernel instrumentation.

    Three tiers of instrumentation, activated based on arguments:

    Tier 1 (instrument_triton=True): Intercepts Triton kernels compiled via
        torch.compile / torch.inductor by delegating to triton_hook.install().

    Tier 2 (instrument_extensions=True): Monkey-patches
        torch.utils.cpp_extension.load_inline to inject the PRLX LLVM pass
        into CUDA compilation flags for custom C++/CUDA extensions.

    Tier 3 (nvbit_precompiled=True): Configures LD_PRELOAD with
        libprlx_nvbit.so for SASS-level instrumentation of pre-compiled
        kernels (cuBLAS, cuDNN, etc.).

    Args:
        instrument_triton: Enable Tier 1 Triton instrumentation.
        instrument_extensions: Enable Tier 2 load_inline hook.
        nvbit_precompiled: Enable Tier 3 NVBit fallback.
        verbose: Print status messages to stderr.
    """
    global _installed

    if _installed:
        logger.info("PyTorch integration already installed")
        return

    logger.info("Installing PyTorch integration...")

    tiers_installed = 0

    if instrument_triton:
        try:
            _hook_triton_via_torch(verbose)
            tiers_installed += 1
        except Exception as e:
            logger.info("Tier 1 (Triton via torch.compile) skipped: %s", e)

    if instrument_extensions:
        try:
            _hook_cpp_extension(verbose)
            tiers_installed += 1
        except Exception as e:
            logger.info("Tier 2 (load_inline hook) skipped: %s", e)

    if nvbit_precompiled:
        try:
            _setup_nvbit(verbose)
            tiers_installed += 1
        except Exception as e:
            logger.info("Tier 3 (NVBit fallback) skipped: %s", e)

    if tiers_installed == 0:
        raise RuntimeError(
            "No instrumentation tiers could be activated. "
            "Check that required libraries are built and available."
        )

    _installed = True
    logger.info("PyTorch integration installed (%d tier(s) active)", tiers_installed)


def _hook_triton_via_torch(verbose: bool):
    """Tier 1: Delegate to triton_hook.install() for torch.compile / inductor.

    When PyTorch uses torch.compile, it lowers operations through
    torch.inductor which generates Triton kernels. Our existing
    triton_hook intercepts the LLVM IR stage of Triton compilation,
    so this tier simply activates it.
    """
    global _tier1_active

    # Verify Triton is available (torch.compile needs it)
    try:
        import triton  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Triton not found. Install with: pip install triton>=3.0"
        )

    from .triton_hook import install as triton_install

    triton_install(verbose=verbose)
    _tier1_active = True
    logger.info(
        "Tier 1: Triton instrumentation active "
        "(covers torch.compile / inductor kernels)"
    )


def _hook_cpp_extension(verbose: bool):
    """Tier 2: Monkey-patch load_inline to inject PRLX pass into CUDA builds.

    PyTorch's torch.utils.cpp_extension.load_inline compiles custom
    C++/CUDA extensions at runtime. We intercept this to append
    -fpass-plugin=libPrlxPass.so to extra_cuda_cflags, so any CUDA
    kernels compiled through this path get instrumented.
    """
    global _original_load_inline, _tier2_active

    pass_plugin = find_pass_plugin()
    if pass_plugin is None or not pass_plugin.exists():
        raise FileNotFoundError(
            "libPrlxPass.so not found. Build with: cmake --build build"
        )

    runtime_lib = find_runtime_library()
    if runtime_lib is None or not runtime_lib.exists():
        raise FileNotFoundError(
            "libprlx_runtime_shared.so not found. "
            "Build with: cmake --build build --target prlx_runtime_shared"
        )

    import torch.utils.cpp_extension as cpp_ext

    if _original_load_inline is not None:
        # Already hooked
        _tier2_active = True
        return

    _original_load_inline = cpp_ext.load_inline

    pass_flag = f"-fpass-plugin={pass_plugin}"
    runtime_link_flag = f"-L{runtime_lib.parent}"
    runtime_lib_flag = "-lprlx_runtime_shared"

    @functools.wraps(_original_load_inline)
    def patched_load_inline(*args, **kwargs):
        if os.environ.get("PRLX_ENABLED", "1") == "0":
            return _original_load_inline(*args, **kwargs)

        # Inject pass plugin into extra_cuda_cflags
        cuda_cflags = list(kwargs.get("extra_cuda_cflags", None) or [])
        if pass_flag not in cuda_cflags:
            cuda_cflags.append(pass_flag)
        kwargs["extra_cuda_cflags"] = cuda_cflags

        # Inject runtime library into extra_ldflags
        ldflags = list(kwargs.get("extra_ldflags", None) or [])
        if runtime_link_flag not in ldflags:
            ldflags.append(runtime_link_flag)
        if runtime_lib_flag not in ldflags:
            ldflags.append(runtime_lib_flag)
        kwargs["extra_ldflags"] = ldflags

        return _original_load_inline(*args, **kwargs)

    cpp_ext.load_inline = patched_load_inline
    _tier2_active = True
    logger.info("Tier 2: load_inline hooked (pass: %s)", pass_plugin)


def _setup_nvbit(verbose: bool):
    """Tier 3: Configure LD_PRELOAD for NVBit SASS-level instrumentation.

    NVBit instruments at the SASS (native GPU assembly) level, so it
    can trace pre-compiled libraries like cuBLAS, cuDNN, and any other
    GPU code that was not compiled with the PRLX LLVM pass.

    WARNING: LD_PRELOAD must be set BEFORE the CUDA driver is
    initialized. If torch.cuda has already been used, NVBit cannot
    intercept existing contexts.
    """
    global _tier3_active

    nvbit_lib = find_nvbit_library()
    if nvbit_lib is None or not nvbit_lib.exists():
        raise FileNotFoundError(
            "libprlx_nvbit.so not found. Build with: cmake --build build --target prlx_nvbit"
        )

    # Warn if CUDA context is already initialized
    try:
        import torch.cuda
        if torch.cuda.is_available() and torch.cuda._initialized:
            logger.warning(
                "CUDA already initialized. NVBit LD_PRELOAD "
                "may not intercept existing contexts. For best results, call "
                "prlx.enable_pytorch(nvbit_precompiled=True) before any "
                "torch.cuda operations."
            )
    except (ImportError, AttributeError):
        pass  # torch.cuda not available or no _initialized attr

    # Set LD_PRELOAD
    current_preload = os.environ.get("LD_PRELOAD", "")
    nvbit_str = str(nvbit_lib)
    if nvbit_str not in current_preload:
        if current_preload:
            os.environ["LD_PRELOAD"] = f"{nvbit_str}:{current_preload}"
        else:
            os.environ["LD_PRELOAD"] = nvbit_str

    _tier3_active = True
    logger.info("Tier 3: NVBit LD_PRELOAD configured (%s)", nvbit_lib)
    if not current_preload:
        logger.info(
            "NOTE: LD_PRELOAD was set in this process. If the CUDA "
            "driver was already loaded, you may need to re-launch the "
            "program with:\n  LD_PRELOAD=%s python your_script.py",
            nvbit_str,
        )


class PrlxTorchWrapper:
    """Context manager for tracing PyTorch GPU operations.

    Wraps a block of PyTorch code with PRLX session tracing, so that
    all GPU kernel launches within the block are captured.

    Usage::

        with prlx.pytorch_trace("my_model_forward") as wrapper:
            output = model(input)

        # Traces are written to PRLX_TRACE directory
    """

    def __init__(
        self,
        name: str,
        output: Optional[str] = None,
        session: Optional[str] = None,
    ):
        self.name = name
        self.output = output
        self.session = session or name
        self._runtime = None
        self._prev_trace = None
        self._prev_session = None

    def __enter__(self):
        # Set output directory if specified
        if self.output:
            self._prev_trace = os.environ.get("PRLX_TRACE")
            os.environ["PRLX_TRACE"] = self.output

        # Set session name
        self._prev_session = os.environ.get("PRLX_SESSION")
        os.environ["PRLX_SESSION"] = self.session

        # Start a runtime session
        rt_lib = find_runtime_library()
        if rt_lib is not None:
            from .runtime import PrlxRuntime
            self._runtime = PrlxRuntime(str(rt_lib))
            self._runtime.session_begin(self.session)

        # Synchronize GPU before tracing
        try:
            import torch.cuda
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass  # no GPU or torch.cuda unavailable

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Synchronize GPU to capture all pending kernels
        try:
            import torch.cuda
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass

        # End runtime session
        if self._runtime is not None:
            self._runtime.session_end()
            self._runtime = None

        # Restore environment
        if self._prev_trace is not None:
            os.environ["PRLX_TRACE"] = self._prev_trace
        elif self.output and "PRLX_TRACE" in os.environ:
            del os.environ["PRLX_TRACE"]

        if self._prev_session is not None:
            os.environ["PRLX_SESSION"] = self._prev_session
        elif "PRLX_SESSION" in os.environ:
            del os.environ["PRLX_SESSION"]

        # Do not suppress exceptions
        return False


def uninstall():
    """Remove all PyTorch integration hooks."""
    global _installed, _original_load_inline, _tier1_active, _tier2_active, _tier3_active

    # Tier 1: uninstall Triton hook
    if _tier1_active:
        try:
            from .triton_hook import uninstall as triton_uninstall
            triton_uninstall()
        except Exception as e:
            logger.debug("Failed to uninstall Triton hook: %s", e)
        _tier1_active = False

    # Tier 2: restore original load_inline
    if _tier2_active and _original_load_inline is not None:
        try:
            import torch.utils.cpp_extension as cpp_ext
            cpp_ext.load_inline = _original_load_inline
        except ImportError:
            pass  # torch not available
        _original_load_inline = None
        _tier2_active = False

    # Tier 3: clean LD_PRELOAD
    if _tier3_active:
        nvbit_lib = find_nvbit_library()
        if nvbit_lib is not None:
            current_preload = os.environ.get("LD_PRELOAD", "")
            nvbit_str = str(nvbit_lib)
            parts = [p for p in current_preload.split(":") if p != nvbit_str]
            if parts:
                os.environ["LD_PRELOAD"] = ":".join(parts)
            elif "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
        _tier3_active = False

    _installed = False
