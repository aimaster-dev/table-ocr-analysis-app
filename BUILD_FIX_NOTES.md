# PyInstaller build fix

This revision addresses the Windows PyInstaller recursion failure reported while
building Table Scan with Python 3.10.11, PyInstaller 6.21.0, PaddlePaddle 3.2.2,
PaddleOCR 3.6.0, and PaddleX 3.6.0.

## Changes

- Raises the analysis recursion limit inside `build.spec` before `Analysis` runs.
- Stops force-collecting every PaddleOCR and PaddleX submodule.
- Excludes unrelated optional TensorRT, serving, Doc2MD, test, and benchmark trees.
- Preserves required OCR package data, distribution metadata, and Paddle native libraries.
- Adds a frozen-runtime hook that sanitizes Paddle's site-package lookup and exposes
  the bundled `paddle/libs` directory to the Windows DLL loader.
- Adds strict Python/package preflight checks to `scripts/build_exe.ps1`.
- Adds a post-build frozen import smoke test and report.

## Build

From the activated Python 3.10 virtual environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

The script expects the tested versions pinned in `requirements.txt`. A successful
run creates `dist\TableScan\TableScan.exe` and
`build\packaging-smoke-test.txt`. Distribute the entire `dist\TableScan` folder.
