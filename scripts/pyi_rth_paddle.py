"""PyInstaller runtime compatibility for PaddlePaddle on Windows.

Paddle 3.x discovers ``paddle/libs`` through ``site.getsitepackages()``.
Some frozen environments expose a ``None`` entry there, which causes Paddle's
path join to fail before its native runtime is imported. This hook runs before
application imports, removes invalid entries, and exposes the bundled native
DLL directory to Windows' loader.

It also patches PaddleX's metadata-based dependency checks so OCR-core extras
that are bundled as modules (but may lack dist-info in older builds) still pass.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import site
import sys

_DLL_DIRECTORY_HANDLES: list[object] = []

# importlib.metadata package name → importable module name in the frozen app
_FROZEN_DEP_MODULES = {
    "opencv-contrib-python": "cv2",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "python-bidi": "bidi",
    "pillow": "PIL",
    "pyyaml": "yaml",
}


def _as_path_string(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = os.fspath(value)
    except TypeError:
        return None
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    return text if isinstance(text, str) and text else None


def _unique_existing_roots(values: list[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_path_string(value)
        if not text:
            continue
        normalized = os.path.normcase(os.path.abspath(text))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result


def _patch_paddlex_dependency_checks() -> None:
    """Allow OCR when modules exist even if dist-info metadata is incomplete."""
    try:
        from paddlex.utils import deps
    except Exception:  # noqa: BLE001
        return

    original = deps.is_dep_available

    def _module_available(dep: str) -> bool:
        module_name = _FROZEN_DEP_MODULES.get(dep, dep.replace("-", "_"))
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def is_dep_available(dep, /, check_version=False):  # noqa: ANN001
        if original(dep, check_version=check_version):
            return True
        # Metadata can be missing in PyInstaller bundles even when the code is
        # present. Fall back to module discovery for frozen apps only.
        return _module_available(dep)

    deps.is_dep_available = is_dep_available
    if hasattr(deps.is_extra_available, "cache_clear"):
        deps.is_extra_available.cache_clear()


if getattr(sys, "frozen", False):
    original_getsitepackages = getattr(site, "getsitepackages", lambda: [])
    try:
        original_roots = list(original_getsitepackages() or [])
    except Exception:  # noqa: BLE001 - startup compatibility guard
        original_roots = []

    bundle_root = _as_path_string(getattr(sys, "_MEIPASS", None))
    executable_root = str(Path(sys.executable).resolve().parent)
    candidate_roots: list[object] = [
        bundle_root,
        executable_root,
        str(Path(executable_root) / "_internal"),
        *original_roots,
    ]
    sanitized_roots = _unique_existing_roots(candidate_roots)

    def _frozen_getsitepackages() -> list[str]:
        return list(sanitized_roots)

    site.getsitepackages = _frozen_getsitepackages

    # ``site.USER_SITE`` can also be None in a frozen interpreter. Paddle only
    # needs a valid string or a false-y value here.
    if _as_path_string(getattr(site, "USER_SITE", None)) is None:
        site.USER_SITE = ""

    for root in sanitized_roots:
        paddle_libs = Path(root) / "paddle" / "libs"
        if not paddle_libs.is_dir():
            continue
        libs_text = str(paddle_libs)
        current_path = os.environ.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if libs_text not in path_parts:
            os.environ["PATH"] = libs_text + (
                os.pathsep + current_path if current_path else ""
            )
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(libs_text))
            except OSError:
                pass

    # paddlex may not be importable yet at hook time on some builds.
    try:
        _patch_paddlex_dependency_checks()
    except Exception:  # noqa: BLE001
        pass
