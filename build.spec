# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Table Scan (Windows onedir .exe)."""

from __future__ import annotations

import sys
from pathlib import Path

# Paddle/PaddleX has a deep import graph. PyInstaller's module-graph traversal is
# recursive, so the default Python recursion limit is too small for this build.
# The limit must be raised in the spec itself because a spec file is executable
# Python and is evaluated before Analysis starts.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

from PyInstaller.utils.hooks import (  # noqa: E402
    collect_all,
    collect_data_files,
    copy_metadata,
)

block_cipher = None
project_root = Path(SPECPATH).resolve()
src_root = project_root / "src"
styles_dir = src_root / "table_scan" / "ui" / "styles"
runtime_hook = project_root / "scripts" / "pyi_rth_paddle.py"


def _is_test_module(name: str) -> bool:
    """Return True for package test/benchmark modules that are not runtime code."""
    parts = name.lower().split(".")
    return any(
        part in {"test", "tests", "testing", "benchmark", "benchmarks"}
        or part.startswith("test_")
        for part in parts
    )


def _keep_paddle_submodule(name: str) -> bool:
    """Keep Paddle runtime modules while skipping known optional/problem trees."""
    if _is_test_module(name):
        return False
    # CPU OCR does not use TensorRT. Importing this optional tree during hook
    # discovery raises against CPU builds of libpaddle and is unnecessary.
    if name == "paddle.tensorrt" or name.startswith("paddle.tensorrt."):
        return False
    return True


def _dedupe(items: list) -> list:
    """Preserve order while removing duplicate PyInstaller TOC entries."""
    seen = set()
    result = []
    for item in items:
        key = tuple(item) if isinstance(item, (tuple, list)) else item
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


datas = [
    (str(styles_dir / "theme.qss"), "table_scan/ui/styles"),
]
binaries: list = []
hiddenimports: list[str] = [
    "pytesseract",
    "cv2",
    "numpy",
    "PIL",
    "openpyxl",
    "paddle",
    "paddleocr",
    "paddlex",
    "paddleocr._pipelines.ocr",
    "paddlex.inference.pipelines.ocr",
    "paddlex.inference.models.text_detection",
    "paddlex.inference.models.text_recognition",
    "paddlex.inference.models.image_classification",
    "paddlex.inference.models.image_unwarping",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Paddle contains native DLLs and several dynamically imported modules. Keep
# its runtime graph, but do not force PyInstaller to analyze tests/TensorRT.
try:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(
        "paddle",
        include_py_files=False,
        filter_submodules=_keep_paddle_submodule,
        on_error="warn once",
    )
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden
except Exception as exc:  # noqa: BLE001
    print(f"[build.spec] collect_all(paddle) failed: {exc}")

# PaddleOCR 3.x installs PaddleX and both packages ship required YAML/model
# metadata/fonts as package data. Their Python imports are already reachable
# from the package roots; collecting *every* submodule is what pulled optional
# serving/doc2md/gen-AI stacks into the old build and triggered RecursionError.
for package in ("paddleocr", "paddlex"):
    try:
        datas += collect_data_files(package, include_py_files=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[build.spec] collect_data_files({package}) failed: {exc}")

# PaddleX and PaddleOCR query package metadata at runtime for version and
# optional-dependency checks. Preserve the dist-info records in the bundle.
for distribution in ("paddlepaddle", "paddleocr", "paddlex"):
    try:
        datas += copy_metadata(distribution, recursive=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[build.spec] copy_metadata({distribution}) failed: {exc}")

datas = _dedupe(datas)
binaries = _dedupe(binaries)
hiddenimports = list(dict.fromkeys(hiddenimports))

# These optional trees are discovered by broad third-party package imports but
# are not used by Table Scan's local CPU OCR flow. Excluding them avoids their
# missing-extra warnings and keeps Analysis from traversing unrelated servers,
# document converters, and package test suites.
excludes = [
    "tkinter",
    "matplotlib",
    "torch",
    "torchvision",
    "easyocr",
    "paddle.tensorrt",
    "paddleocr._doc2md",
    "paddlex.inference.serving",
    "pytest",
    "_pytest",
]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(src_root), str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(runtime_hook)],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TableScan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TableScan",
)
