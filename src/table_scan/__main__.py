"""CLI / module entry: ``python -m table_scan``."""

from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import sys
import traceback


def _packaging_smoke_test(report_path: str | None) -> int:
    """Verify that core frozen dependencies can be imported without opening Qt."""
    lines = [
        "Table Scan packaging smoke test",
        f"frozen={bool(getattr(sys, 'frozen', False))}",
        f"executable={sys.executable}",
        f"bundle_root={getattr(sys, '_MEIPASS', '')}",
    ]
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import openpyxl  # noqa: F401
        import PySide6.QtCore  # noqa: F401
        import PySide6.QtWidgets  # noqa: F401
        import paddle  # noqa: F401
        import paddlex  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401
        from table_scan.core.html_writer import HtmlExporter  # noqa: F401

        for distribution in (
            "table-scan",
            "paddlepaddle",
            "paddleocr",
            "paddlex",
            "PySide6",
            "opencv-python-headless",
            "openpyxl",
        ):
            try:
                lines.append(f"{distribution}={metadata.version(distribution)}")
            except metadata.PackageNotFoundError:
                # table-scan's own dist-info is not required in a PyInstaller app.
                lines.append(f"{distribution}=metadata unavailable")

        if getattr(sys, "frozen", False):
            roots = [
                Path(str(getattr(sys, "_MEIPASS", ""))),
                Path(sys.executable).resolve().parent,
                Path(sys.executable).resolve().parent / "_internal",
            ]
            paddle_lib_dirs = [root / "paddle" / "libs" for root in roots]
            existing = [str(path) for path in paddle_lib_dirs if path.is_dir()]
            if not existing:
                raise RuntimeError(
                    "Bundled paddle/libs directory was not found. Checked: "
                    + ", ".join(str(path) for path in paddle_lib_dirs)
                )
            lines.append("paddle_libs=" + "; ".join(existing))

        lines.append("STATUS: OK")
        exit_code = 0
    except Exception:  # noqa: BLE001 - this is a diagnostic boundary
        lines.append("STATUS: FAILED")
        lines.append(traceback.format_exc())
        exit_code = 1

    output = "\n".join(lines) + "\n"
    if report_path:
        try:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output, encoding="utf-8")
        except OSError:
            return 2
    else:
        print(output)
    return exit_code


def main() -> int:
    if "--packaging-smoke-test" in sys.argv:
        index = sys.argv.index("--packaging-smoke-test")
        report_path = (
            sys.argv[index + 1]
            if index + 1 < len(sys.argv)
            else os.environ.get("TABLESCAN_SMOKE_REPORT")
        )
        return _packaging_smoke_test(report_path)

    from table_scan.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
