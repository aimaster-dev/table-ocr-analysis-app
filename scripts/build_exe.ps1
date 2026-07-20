# Build Table Scan into a Windows folder distribution (dist\TableScan\TableScan.exe)
# Usage (from project root, in an activated venv):
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "==> Project root: $Root"
Write-Host "==> Python: $Python"

# Keep the build isolated from packages installed in the user's global site.
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

# Ensure PyInstaller is available in this interpreter.
& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Installing PyInstaller..."
    & $Python -m pip install "pyinstaller==6.21.0"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller"
    }
}

Write-Host "==> Validating the build environment..."
$Preflight = @'
import importlib.metadata as md
import platform
import struct
import sys

required = {
    "paddlepaddle": "3.2.2",
    "paddleocr": "3.6.0",
    "paddlex": "3.6.0",
}
errors = []
for package, expected in required.items():
    try:
        actual = md.version(package)
    except md.PackageNotFoundError:
        errors.append(f"{package} is not installed")
        continue
    if actual != expected:
        errors.append(f"{package} must be {expected}, found {actual}")

if struct.calcsize("P") * 8 != 64:
    errors.append("A 64-bit Python interpreter is required")
if sys.version_info[:2] != (3, 10):
    errors.append(
        f"The tested build uses Python 3.10; found {sys.version_info.major}.{sys.version_info.minor}"
    )

if errors:
    print("Build preflight failed:")
    for error in errors:
        print(f"  - {error}")
    print()
    print('Repair command:')
    print('  python -m pip install --upgrade --force-reinstall "paddlepaddle==3.2.2" "paddleocr==3.6.0" "paddlex==3.6.0" "pyinstaller==6.21.0"')
    raise SystemExit(2)

import paddle
import paddleocr
import paddlex
print(f"Python {platform.python_version()} ({platform.architecture()[0]})")
print(f"paddlepaddle={md.version('paddlepaddle')}")
print(f"paddleocr={md.version('paddleocr')}")
print(f"paddlex={md.version('paddlex')}")
print(f"pyinstaller={md.version('pyinstaller')}")
'@

& $Python -c $Preflight
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build environment validation failed"
}

Write-Host "==> Cleaning previous build artifacts..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .\build, .\dist

Write-Host "==> Running PyInstaller..."
& $Python -m PyInstaller --noconfirm --clean build.spec
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed with exit code $LASTEXITCODE"
}

$Exe = Join-Path $Root "dist\TableScan\TableScan.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build finished but executable was not found at $Exe"
}

Write-Host "==> Running frozen import smoke test..."
$SmokeReport = Join-Path $Root "build\packaging-smoke-test.txt"
Remove-Item -Force -ErrorAction SilentlyContinue $SmokeReport
$env:TABLESCAN_SMOKE_REPORT = $SmokeReport
try {
    $SmokeProcess = Start-Process `
        -FilePath $Exe `
        -ArgumentList @("--packaging-smoke-test") `
        -Wait `
        -PassThru
} finally {
    Remove-Item Env:\TABLESCAN_SMOKE_REPORT -ErrorAction SilentlyContinue
}

if ($SmokeProcess.ExitCode -ne 0 -or -not (Test-Path $SmokeReport)) {
    if (Test-Path $SmokeReport) {
        Write-Host ""
        Get-Content $SmokeReport
    }
    Write-Error "Frozen import smoke test failed with exit code $($SmokeProcess.ExitCode)"
}

$SmokeText = Get-Content $SmokeReport -Raw
if ($SmokeText -notmatch "STATUS: OK") {
    Write-Host ""
    Write-Host $SmokeText
    Write-Error "Frozen import smoke test did not report success"
}

Write-Host ""
Write-Host "Build succeeded:"
Write-Host "  $Exe"
Write-Host ""
Write-Host "Smoke test report:"
Write-Host "  $SmokeReport"
Write-Host ""
Write-Host "Distribute the entire dist\TableScan folder (not only the .exe)."
