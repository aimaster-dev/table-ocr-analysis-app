"""Resolve Tesseract install directory / executable / tessdata paths."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")


def resolve_tesseract_cmd(location: str | Path) -> Path:
    """
    Accept either:
    - path to ``tesseract.exe``
    - path to the Tesseract install directory (contains tesseract.exe)
    - path to a ``tessdata`` folder (parent is the install directory)
    """
    path = Path(location).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"OCR engine path not found:\n{path}")

    if path.is_file():
        if path.name.lower() == "tesseract.exe" or path.suffix.lower() == ".exe":
            return path
        raise FileNotFoundError(f"Not a Tesseract executable:\n{path}")

    # Directory: prefer tesseract.exe inside it.
    candidate = path / "tesseract.exe"
    if candidate.is_file():
        return candidate

    # tessdata folder selected → parent install dir.
    if path.name.lower() == "tessdata":
        candidate = path.parent / "tesseract.exe"
        if candidate.is_file():
            return candidate

    # Nested common layout.
    nested = path / "Tesseract-OCR" / "tesseract.exe"
    if nested.is_file():
        return nested

    raise FileNotFoundError(
        "Could not find tesseract.exe under:\n"
        f"{path}\n\n"
        "Select the Tesseract-OCR install folder or tesseract.exe itself."
    )


def resolve_tessdata_dir(location: str | Path) -> Path | None:
    """Return tessdata directory if present next to the resolved executable."""
    try:
        cmd = resolve_tesseract_cmd(location)
    except FileNotFoundError:
        return None

    for candidate in (cmd.parent / "tessdata", cmd.parent.parent / "tessdata"):
        if candidate.is_dir():
            return candidate
    return None


def apply_tessdata_env(location: str | Path) -> None:
    """Set TESSDATA_PREFIX so Tesseract finds language models."""
    tessdata = resolve_tessdata_dir(location)
    if tessdata is not None:
        # Tesseract expects the parent of tessdata, or tessdata itself depending on version.
        # Setting to the tessdata folder works with UB Mannheim builds when suffix is correct.
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
