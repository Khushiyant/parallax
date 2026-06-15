"""Library and binary discovery for prlx components.

Search order: PRLX_HOME env -> bundled package data -> project root -> system PATH.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

_PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def _project_root() -> Optional[Path]:
    """Walk up to find the source tree root (has CMakeLists.txt + differ/)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        current = current.parent
        if (current / "CMakeLists.txt").exists() and (current / "differ").exists():
            return current
    return None


def _detect_llvm_version() -> Optional[int]:
    opt = find_opt_binary()
    if opt is None:
        return None
    try:
        result = subprocess.run(
            [str(opt), "--version"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            m = re.search(r"(\d+)\.\d+", line)
            if m:
                return int(m.group(1))
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def find_pass_plugin() -> Optional[Path]:
    """Find libPrlxPass.so, matching the system LLVM version for bundled installs."""
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libPrlxPass.so",
            Path(home) / "build" / "lib" / "pass" / "libPrlxPass.so",
        ]:
            if p.exists():
                return p.resolve()

    # Bundled: pick version-matched .so
    llvm_ver = _detect_llvm_version()
    if llvm_ver:
        p = _PACKAGE_DATA / "lib" / f"libPrlxPass.llvm{llvm_ver}.so"
        if p.exists():
            return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libPrlxPass.so"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "pass" / "libPrlxPass.so"
            if p.exists():
                return p.resolve()

    return None


def find_runtime_library() -> Optional[Path]:
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libprlx_runtime_shared.so",
            Path(home) / "build" / "lib" / "runtime" / "libprlx_runtime_shared.so",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libprlx_runtime_shared.so"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "libprlx_runtime_shared.so"
            if p.exists():
                return p.resolve()

    return None


def find_static_runtime() -> Optional[Path]:
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libprlx_runtime.a",
            Path(home) / "build" / "lib" / "runtime" / "libprlx_runtime.a",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libprlx_runtime.a"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "libprlx_runtime.a"
            if p.exists():
                return p.resolve()

    return None


def find_runtime_bitcode() -> Optional[Path]:
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "prlx_runtime_nvptx.bc",
            Path(home) / "build" / "lib" / "runtime" / "prlx_runtime_nvptx.bc",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "prlx_runtime_nvptx.bc"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "prlx_runtime_nvptx.bc"
            if p.exists():
                return p.resolve()

    return None


def find_include_dirs() -> List[Path]:
    inc = _PACKAGE_DATA / "include"
    if inc.exists() and (inc / "trace_format.h").exists():
        return [inc]

    root = _project_root()
    if root:
        dirs = []
        for d in (root / "lib" / "runtime", root / "lib" / "common"):
            if d.exists():
                dirs.append(d)
        if dirs:
            return dirs

    return []


def find_opt_binary() -> Optional[Path]:
    for name in ("opt-20", "opt-19", "opt-18", "opt"):
        path = shutil.which(name)
        if path:
            return Path(path)
    # apt.llvm.org installs to /usr/lib/llvm-N/bin without adding it to PATH
    for v in (20, 19, 18):
        p = Path(f"/usr/lib/llvm-{v}/bin/opt")
        if p.exists():
            return p
    return None


def find_llvm_link_binary() -> Optional[Path]:
    for name in ("llvm-link-20", "llvm-link-19", "llvm-link-18", "llvm-link"):
        path = shutil.which(name)
        if path:
            return Path(path)
    for v in (20, 19, 18):
        p = Path(f"/usr/lib/llvm-{v}/bin/llvm-link")
        if p.exists():
            return p
    return None


def find_nvbit_library() -> Optional[Path]:
    """Find libprlx_nvbit.so for SASS-level binary instrumentation."""
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libprlx_nvbit.so",
            Path(home) / "build" / "lib" / "nvbit_tool" / "libprlx_nvbit.so",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libprlx_nvbit.so"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "nvbit_tool" / "libprlx_nvbit.so"
            if p.exists():
                return p.resolve()

    return None


def find_differ_binary() -> Optional[Path]:
    home = os.environ.get("PRLX_HOME")
    if home:
        p = Path(home) / "bin" / "prlx-diff"
        if p.exists():
            return p.resolve()

    p = _PACKAGE_DATA / "bin" / "prlx-diff"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for variant in ("release", "debug"):
            p = root / "differ" / "target" / variant / "prlx-diff"
            if p.exists():
                return p.resolve()

    path = shutil.which("prlx-diff")
    if path:
        return Path(path)

    return None
