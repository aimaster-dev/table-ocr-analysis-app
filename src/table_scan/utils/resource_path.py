"""Path helpers that work in both development and frozen (PyInstaller) builds."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def app_root() -> Path:
    """Project / bundle root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    """``table_scan`` package directory (or extracted MEIPASS copy)."""
    if is_frozen():
        return Path(sys._MEIPASS) / "table_scan"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Resolve a file shipped inside the package (styles, icons, …)."""
    return package_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Writable per-user data directory for settings and caches."""
    base = Path.home() / "AppData" / "Local" / "TableScan"
    base.mkdir(parents=True, exist_ok=True)
    return base
