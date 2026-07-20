"""Application entry point for development and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script (dev mode).
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from table_scan.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
