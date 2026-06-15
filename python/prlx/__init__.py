"""PRLX - Differential debugger for CUDA/Triton GPU kernels."""

import logging as _logging

_logger = _logging.getLogger(__name__)

try:
    from importlib.metadata import version as _meta_version
    __version__ = _meta_version("prlx")
except Exception as _e:
    __version__ = "0.0.0.dev0"
    _logger.debug("Could not read prlx version from metadata: %s", _e)

from .trace_reader import read_trace, TraceData, TraceEvent, TraceHeader, WarpData
from .training_monitor import PrlxTrainingMonitor
from ._find_lib import find_differ_binary


def enable(**kwargs):
    """Hook into Triton's compilation pipeline to instrument kernels."""
    from .triton_hook import install
    install(**kwargs)


def integrate_with_triton(**kwargs):
    """Alias for enable()."""
    enable(**kwargs)


def diff_traces(trace_a: str, trace_b: str, **kwargs) -> int:
    """Run the Rust differ on two trace files. Returns exit code (0=identical, 1=divergent)."""
    import subprocess

    differ = find_differ_binary()
    if differ is None:
        raise FileNotFoundError("prlx-diff binary not found. Install with: pip install prlx")

    cmd = [str(differ), str(trace_a), str(trace_b)]

    if kwargs.get("map"):
        cmd.extend(["--map", str(kwargs["map"])])
    if kwargs.get("values"):
        cmd.append("--values")
    if kwargs.get("tui"):
        cmd.append("--tui")
    if kwargs.get("history"):
        cmd.append("--history")

    return subprocess.call(cmd)


def session(name: str, **kwargs):
    """Context manager for multi-kernel pipeline tracing."""
    from .runtime import session as _session
    return _session(name, **kwargs)


def enable_pytorch(**kwargs):
    """Hook into PyTorch for GPU kernel instrumentation.

    Three-tier strategy:
      Tier 1, Triton via torch.compile (instrument_triton=True)
      Tier 2, load_inline hook (instrument_extensions=True)
      Tier 3, NVBit fallback (nvbit_precompiled=False)
    """
    from .pytorch_hook import install
    install(**kwargs)


def pytorch_trace(name: str, **kwargs):
    """Context manager for tracing PyTorch GPU operations.

    Usage::

        with prlx.pytorch_trace("forward_pass") as t:
            output = model(input)
    """
    from .pytorch_hook import PrlxTorchWrapper
    return PrlxTorchWrapper(name, **kwargs)


def plot_warp_timeline(trace, **kwargs):
    """Scatter plot of events by warp (lazy import of matplotlib)."""
    from .viz import plot_warp_timeline as _impl
    return _impl(trace, **kwargs)


def plot_divergence_heatmap(trace_a, trace_b, **kwargs):
    """Heatmap of per-warp divergence counts (lazy import of matplotlib)."""
    from .viz import plot_divergence_heatmap as _impl
    return _impl(trace_a, trace_b, **kwargs)


def display_trace_summary(trace):
    """HTML table summarising a trace (works in Jupyter)."""
    from .viz import display_trace_summary as _impl
    return _impl(trace)
