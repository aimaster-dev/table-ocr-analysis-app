# Table Scan

Desktop app that converts photographed paper tables into Excel (`.xlsx`),
self-contained HTML (`.html`), or both.

## Features

- Select a folder of table photos (PNG, JPG, BMP, TIFF, WEBP)
- Perspective correction, line-aware deskew, and contrast enhancement that
  preserve fine handwritten CJK strokes
- Local table-region detection for off-center, broken-rule, and multi-table photos
- One page-level OCR pass mapped into cells, with focused retries only for weak cells
- **Tesseract** — printed Latin tables
- **PP-OCRv6 mixed handwriting mode** — Korean + Simplified/Traditional Chinese
  + English + numbers in one recognition model, offline after first download
- **Local VLM (Ollama)** — hard handwriting / messy photos, offline
- Excel, HTML, or both from one OCR pass
- One workbook/document per image, including multiple detected tables
- Native **Qt (PySide6)** UI
- Windows **`.exe`** packaging via PyInstaller

## Requirements

- Python 3.10+
- Optional engines:
  - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  - PaddleOCR: `pip install "paddlepaddle==3.2.2" "paddleocr==3.6.0"`
  - [Ollama](https://ollama.com) + a vision model (for VLM mode)

## Mixed Korean / Chinese handwriting (recommended)

```powershell
.\.venv\Scripts\Activate.ps1
pip install "paddlepaddle==3.2.2" "paddleocr==3.6.0"
```

In the app:

1. OCR engine → **Local PaddleOCR (East Asian / handwriting)**
2. Language → **Handwritten Korean + Chinese + English / numbers (high accuracy)**
3. The first run downloads PP-OCRv6 models; later runs work offline.

This mode deliberately uses one PP-OCRv6 medium recognition model. Running
separate Korean and Chinese recognizers is unsafe when both scripts appear in
one cell because each model may replace the other script with plausible-looking
wrong characters. PP-OCRv6 keeps Korean, Chinese, Latin characters, and digits
in one vocabulary.

Handwriting-specific behavior:

- Low-resolution pages are enlarged with stroke-preserving Lanczos resampling.
- Faint or uneven low-confidence cells receive a second contrast/binary pass.
- Shaky hand-drawn grids have a Hough-geometry fallback.
- Trapezoidal phone photos are rectified before grid detection and OCR.
- Excel cells below 65% OCR confidence are highlighted pale yellow for review.
- Unicode full-width forms are normalized, but leading zeroes, punctuation,
  IDs, units, and numeric ranges are never guessed or rewritten.

The language-specific lightweight modes remain available for faster processing
of documents that contain only one main script.

## Accuracy and performance changes in 1.1+

- OCR runs once over the page instead of running text detection in every cell.
- Recognized page boxes are mapped back to the detected grid; only blank or
  low-confidence cells that actually contain ink are retried.
- Table rules are removed from the OCR view without changing box coordinates.
- Unicode full-width forms are normalized consistently, while IDs, ranges,
  punctuation, leading zeroes, and non-price numeric values are preserved.
- PaddleOCR 2.x and 3.x result formats are both supported.

## Mixed-handwriting changes in 1.2

- Adds PP-OCRv6 medium mixed recognition for Korean, Chinese, English, and numbers.
- Adds high-resolution page inference with coordinate-safe mapping back to cells.
- Adds a conditional handwriting threshold retry rather than doubling every OCR call.
- Detects mildly shaky hand-drawn table rules.
- Corrects strong photographed perspective while rejecting weak/unsafe quadrilaterals.
- Flags uncertain Excel cells instead of hiding low-confidence recognition.

## Windows CPU compatibility (1.2.1)

PaddlePaddle 3.3.x can fail during CPU inference inside its PIR/oneDNN executor
with `ConvertPirAttribute2RuntimeAttribute not support`. Table Scan disables
oneDNN for OCR and pins the stable PaddlePaddle 3.2.2 runtime.

If the environment already contains PaddlePaddle 3.3.x, run this once inside
the activated virtual environment:

```powershell
python -m pip install --upgrade --force-reinstall `
  "paddlepaddle==3.2.2" "paddleocr==3.6.0"
```

The `No ccache found` message is harmless for normal inference. It only concerns
compiling custom Paddle extensions.

## Cropped-grid and spreadsheet screenshot accuracy (1.3)

- Recovers a final row or column whose outer rule is clipped by the image edge.
- Uses reliable morphological rules before Hough geometry, preventing text
  baselines from becoming invented rows and from being erased before OCR.
- Treats missing dividers as merged cells only when neighboring rows confirm
  the merge; faint photographed rules no longer collapse whole records.
- Removes recognizable spreadsheet row/column chrome and empty outer grid bands.
- Preserves OCR line breaks, restores repeated bullet markers, and uses focused
  Tesseract retries for narrow or uncertain cells.
- Expands wrapped Excel rows and hides redundant default gridlines so extracted
  content is visible immediately.

## HTML output (1.4)

- Adds an **Output format** selector: Excel, HTML, or Excel + HTML.
- Both formats consume the same normalized `ExtractedTable`; selecting Both does
  not rerun OCR and cannot introduce format-specific recognition differences.
- HTML preserves Unicode, line breaks, confidence review highlighting, and
  detected `rowspan` / `colspan` merges.
- Generated documents are self-contained, responsive, printable, and safe for
  untrusted OCR text through escaping plus a restrictive Content Security Policy.
- New installs default to Both. Existing settings migrate to Excel-only so an
  upgrade does not unexpectedly create additional files.

## Offline VLM (optional)

```powershell
ollama pull qwen2.5vl:7b
```

OCR engine → **Local VLM / Ollama**, URL `http://127.0.0.1:11434`

## Run from source

```powershell
cd D:\code\scan_photo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python main.py
```

## Build Windows EXE

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

Ship the whole `dist\TableScan` folder. Target PCs need the OCR backend you use (Tesseract and/or Paddle models cache and/or Ollama).

## Usage

1. Browse input folder  
2. Choose output folder  
3. Pick OCR engine + language  
4. Choose Excel, HTML, or Both  
5. Convert  

## Notes

| Engine | Best for |
|---|---|
| Tesseract | Printed English / clear grids |
| **PaddleOCR** | **East Asian handwriting & print** |
| Local VLM | Very messy handwriting / complex pages |

Logs: `%LOCALAPPDATA%\TableScan\logs\table_scan.log`


## PyInstaller build diagnostics

The Windows build now performs three safeguards automatically:

1. Raises PyInstaller's analysis recursion limit inside `build.spec`.
2. Avoids force-collecting PaddleOCR/PaddleX optional serving, document-conversion, and test trees.
3. Runs `TableScan.exe --packaging-smoke-test` after the build to verify Qt, OpenCV, Paddle, PaddleX, PaddleOCR, package metadata, and `paddle/libs` from the frozen folder.

Use the normal command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

A successful build writes the smoke-test report to `build\packaging-smoke-test.txt`. Ship the whole `dist\TableScan` directory.
