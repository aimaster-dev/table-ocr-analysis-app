"""PyInstaller runtime compatibility for PaddlePaddle on Windows.

Paddle 3.x discovers ``paddle/libs`` through ``site.getsitepackages()``.
Some frozen environments expose a ``None`` entry there, which causes Paddle's
path join to fail before its native runtime is imported. This hook runs before
application imports, removes invalid entries, and exposes the bundled native
DLL directory to Windows' loader.
"""

from __future__ import annotations

import os
from pathlib import Path
import site
import sys

_DLL_DIRECTORY_HANDLES: list[object] = []


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
            os.environ["PATH"] = libs_text + (os.pathsep + current_path if current_path else "")
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(libs_text))
            except OSError:
                pass
