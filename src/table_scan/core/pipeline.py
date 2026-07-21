"""End-to-end image → table → Excel/HTML pipeline."""

from __future__ import annotations

import logging
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from table_scan.config.settings import (
    OUTPUT_FORMAT_BOTH,
    OUTPUT_FORMAT_EXCEL,
    OUTPUT_FORMAT_HTML,
    AppSettings,
)
from table_scan.core.excel_writer import ExcelExporter
from table_scan.core.html_writer import HtmlExporter
from table_scan.core.layout_mapper import OcrSpan, map_spans_to_grid, parse_ocr_spans
from table_scan.core.ocr_engine import OcrEngine
from table_scan.core.ocr_factory import create_ocr_engine
from table_scan.core.preprocessor import ImagePreprocessor
from table_scan.core.table_detector import TableDetector, TableGrid
from table_scan.models.table_result import ConversionResult, ExtractedTable, JobStatus

logger = logging.getLogger(__name__)


class TableExtractionPipeline:
    """Orchestrates preprocess → detect → OCR → selected output formats."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.preprocessor = ImagePreprocessor(
            deskew=settings.deskew,
            enhance_contrast=settings.enhance_contrast,
            rectify_perspective=settings.rectify_perspective,
        )
        self.detector = TableDetector()
        self.ocr = create_ocr_engine(settings)
        self.exporter = ExcelExporter()
        self.html_exporter = HtmlExporter()

    def warm_up(self) -> None:
        self.ocr.warm_up()

    def convert_image(
        self,
        image_path: Path,
        output_dir: Path,
    ) -> ConversionResult:
        started = time.perf_counter()
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")

            # Each preprocessing option is independent.  Previously disabling
            # deskew accidentally disabled perspective correction and contrast
            # enhancement as well.
            processed = self.preprocessor.process(image)

            # Offline VLM: ask the model for the whole table (best for handwriting).
            if hasattr(self.ocr, "extract_table"):
                table = self.ocr.extract_table(processed)
                table.source_image = image_path
                tables = [table] if table.row_count > 0 and table.col_count > 0 else []
            else:
                grids = self.detector.detect(processed)
                ruled = any(not grid.is_fallback for grid in grids)
                # One page-level OCR call preserves CJK context and is much
                # faster than detecting text independently in every cell.
                ocr_view = (
                    self.detector.remove_ruling_lines(processed) if ruled else processed
                )
                page_spans = parse_ocr_spans(self.ocr.read_text(ocr_view))
                tables = []
                for grid in grids:
                    if grid.is_fallback:
                        table = self._ocr_full_page(page_spans, image_path)
                    else:
                        table = self._ocr_grid(
                            processed, grid, image_path, page_spans=page_spans
                        )
                    table = self._trim_table_edges(table, grid=grid)
                    if self._has_content(table):
                        tables.append(table)

            if not tables:
                raise ValueError("No table content recognized in image")

            output_format = AppSettings.normalize_output_format(
                self.settings.output_format
            )
            output_paths: list[Path] = []
            exported_formats: list[str] = []
            if output_format in {OUTPUT_FORMAT_EXCEL, OUTPUT_FORMAT_BOTH}:
                excel_path = output_dir / f"{image_path.stem}.xlsx"
                self.exporter.export(tables, excel_path)
                output_paths.append(excel_path)
                exported_formats.append("Excel")
            if output_format in {OUTPUT_FORMAT_HTML, OUTPUT_FORMAT_BOTH}:
                html_path = output_dir / f"{image_path.stem}.html"
                self.html_exporter.export(
                    tables,
                    html_path,
                    title=f"{image_path.stem} — extracted tables",
                )
                output_paths.append(html_path)
                exported_formats.append("HTML")

            if not output_paths:
                raise ValueError(f"Unsupported output format: {output_format}")

            elapsed = time.perf_counter() - started
            return ConversionResult(
                image_path=image_path,
                status=JobStatus.SUCCESS,
                tables=tables,
                output_path=output_paths[0],
                output_paths=output_paths,
                message=(
                    f"Exported {len(tables)} table(s) to "
                    + " + ".join(exported_formats)
                ),
                elapsed_seconds=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Conversion failed for %s", image_path)
            return ConversionResult(
                image_path=image_path,
                status=JobStatus.FAILED,
                message=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )

    def _ocr_grid(
        self,
        image_bgr,
        grid: TableGrid,
        source: Path,
        *,
        page_spans: list[OcrSpan],
    ) -> ExtractedTable:
        rows, confidences = map_spans_to_grid(page_spans, grid)
        self._collapse_merged_cells(rows, confidences, grid.merged_ranges)
        merged_cells = self._merged_cell_lookup(grid)
        col_count = grid.cols

        for row_index, row_boxes in enumerate(grid.cells):
            for c, original_box in enumerate(row_boxes):
                merged = merged_cells.get((row_index, c))
                if merged is not None:
                    anchor_row, anchor_col, x, y, w, h = merged
                    if (row_index, c) != (anchor_row, anchor_col):
                        continue
                else:
                    x, y, w, h = original_box
                current = rows[row_index][c]
                current_conf = confidences[row_index][c]
                crop = self._cell_crop(image_bgr, x, y, w, h)
                has_ink = crop.size > 0 and self._has_ink(crop)
                # Page OCR can occasionally read an inpainted rule junction as
                # letters.  Reject it when the original cell interior is blank.
                if current and not has_ink:
                    rows[row_index][c] = ""
                    confidences[row_index][c] = 0.0
                    continue
                narrow_cell = w / max(h, 1) < 1.8
                # Tesseract's inexpensive page pass is fast, but wrapped cells
                # just below 95% often improve with segmentation tailored to a
                # single crop.  Paddle retries are much costlier and already
                # have their own handwriting confidence strategy.
                local_retry = isinstance(self.ocr, OcrEngine) and current_conf < 0.95
                if (
                    current
                    and current_conf >= self.settings.min_cell_confidence
                    and not narrow_cell
                    and not local_retry
                ):
                    continue

                if not has_ink:
                    continue
                raw, conf = self.ocr.read_cell(crop)
                value = OcrEngine.normalize_cell_value(raw, col_index=c, col_count=col_count)
                value = self._preserve_line_layout(value, current)
                if not value:
                    continue
                # Use the focused retry when the page pass missed the cell or
                # when it is at least as credible and no shorter.
                if (
                    not current
                    or conf > current_conf + 0.03
                    or (conf >= current_conf - 0.06 and len(value) > len(current))
                ):
                    rows[row_index][c] = value
                    confidences[row_index][c] = conf

        normalized = [
            [
                OcrEngine.normalize_cell_value(value, col_index=c, col_count=col_count)
                for c, value in enumerate(row)
            ]
            for row in rows
        ]
        return ExtractedTable(
            rows=normalized,
            confidences=confidences,
            source_image=source,
            merged_ranges=list(grid.merged_ranges),
        )

    @staticmethod
    def _preserve_line_layout(value: str, current: str) -> str:
        """Keep page-detected wrapping when a focused retry returns one line."""
        if not value or "\n" in value or "\n" not in current:
            return value
        # Allow a small amount of slack because a corrected retry can add a
        # digit, space, or punctuation that the page pass omitted.
        target_width = max((len(line) for line in current.splitlines()), default=0) + 2
        if target_width < 10:
            return value
        return "\n".join(
            textwrap.wrap(
                value,
                width=target_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    @staticmethod
    def _trim_table_edges(
        table: ExtractedTable,
        *,
        grid: TableGrid | None = None,
    ) -> ExtractedTable:
        """Remove empty outer grid bands and recognizable spreadsheet chrome."""
        width = table.col_count
        if not table.rows or width == 0:
            return table

        rows = [list(row) + [""] * (width - len(row)) for row in table.rows]
        confidences = [
            list(table.confidences[index]) + [0.0] * (width - len(table.confidences[index]))
            if index < len(table.confidences)
            else [0.0] * width
            for index in range(len(rows))
        ]

        nonempty = lambda value: bool((value or "").strip())
        row_start = 0
        row_end = len(rows)
        while row_start < row_end and not any(nonempty(value) for value in rows[row_start]):
            row_start += 1
        while row_end > row_start and not any(nonempty(value) for value in rows[row_end - 1]):
            row_end -= 1
        if row_start == row_end:
            return ExtractedTable(rows=[], source_image=table.source_image)

        populated_columns = [
            column
            for column in range(width)
            if any(nonempty(rows[row][column]) for row in range(row_start, row_end))
        ]
        if not populated_columns:
            return ExtractedTable(rows=[], source_image=table.source_image)
        col_start, col_end = populated_columns[0], populated_columns[-1] + 1

        # A screenshot of a spreadsheet contributes a first band containing
        # the UI column labels A, B, C... .  Remove it only with geometry that
        # shows the band touching the image top while the table begins after a
        # left gutter; this avoids treating an ordinary A/B/C header as chrome.
        if (
            row_start == 0
            and grid is not None
            and grid.cells
            and grid.cells[0]
            and grid.cells[0][0][1] <= 2
            and grid.cells[0][0][0] > 0
            and TableExtractionPipeline._is_spreadsheet_column_header(
                rows[0][col_start:col_end],
                start_column=col_start,
            )
        ):
            row_start += 1

        cropped_rows = [row[col_start:col_end] for row in rows[row_start:row_end]]
        cropped_confidences = [
            row[col_start:col_end] for row in confidences[row_start:row_end]
        ]
        cropped_merges: list[tuple[int, int, int, int]] = []
        for r1, c1, r2, c2 in table.merged_ranges:
            nr1, nr2 = max(r1, row_start), min(r2, row_end - 1)
            nc1, nc2 = max(c1, col_start), min(c2, col_end - 1)
            if nr1 <= nr2 and nc1 <= nc2 and (nr2 > nr1 or nc2 > nc1):
                cropped_merges.append(
                    (nr1 - row_start, nc1 - col_start, nr2 - row_start, nc2 - col_start)
                )

        return ExtractedTable(
            rows=cropped_rows,
            confidences=cropped_confidences,
            source_image=table.source_image,
            merged_ranges=cropped_merges,
        )

    @staticmethod
    def _is_spreadsheet_column_header(values: list[str], *, start_column: int) -> bool:
        if len(values) < 2:
            return False

        def excel_label(index: int) -> str:
            label = ""
            number = index + 1
            while number:
                number, remainder = divmod(number - 1, 26)
                label = chr(ord("A") + remainder) + label
            return label

        normalized = [str(value or "").strip().upper() for value in values]
        expected = [excel_label(start_column + index) for index in range(len(values))]
        plausible = all(
            not value or (value.isalnum() and len(value) <= 3) for value in normalized
        )
        matches = sum(value == target for value, target in zip(normalized, expected))
        required = max(2, (2 * len(values) + 2) // 3)
        return plausible and matches >= required

    @staticmethod
    def _merged_cell_lookup(
        grid: TableGrid,
    ) -> dict[tuple[int, int], tuple[int, int, int, int, int, int]]:
        lookup: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
        for r1, c1, r2, c2 in grid.merged_ranges:
            boxes = [
                grid.cells[row][col]
                for row in range(r1, min(r2 + 1, len(grid.cells)))
                for col in range(c1, min(c2 + 1, len(grid.cells[row])))
            ]
            if not boxes:
                continue
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[0] + box[2] for box in boxes)
            y2 = max(box[1] + box[3] for box in boxes)
            value = (r1, c1, x1, y1, x2 - x1, y2 - y1)
            for row in range(r1, r2 + 1):
                for col in range(c1, c2 + 1):
                    lookup[(row, col)] = value
        return lookup

    @staticmethod
    def _collapse_merged_cells(
        rows: list[list[str]],
        confidences: list[list[float]],
        merged_ranges: list[tuple[int, int, int, int]],
    ) -> None:
        for r1, c1, r2, c2 in merged_ranges:
            fragments: list[tuple[str, float]] = []
            for row in range(r1, min(r2 + 1, len(rows))):
                for col in range(c1, min(c2 + 1, len(rows[row]))):
                    text = rows[row][col]
                    if text:
                        fragments.append((text, confidences[row][col]))
                    rows[row][col] = ""
                    confidences[row][col] = 0.0
            if r1 >= len(rows) or c1 >= len(rows[r1]) or not fragments:
                continue
            combined = fragments[0][0]
            for text, _ in fragments[1:]:
                separator = ""
                if not (
                    combined
                    and text
                    and (
                        TableExtractionPipeline._is_cjk(combined[-1])
                        or TableExtractionPipeline._is_cjk(text[0])
                    )
                ):
                    separator = " "
                combined += separator + text
            weight = sum(max(len(text), 1) for text, _ in fragments)
            confidence = sum(
                conf * max(len(text), 1) for text, conf in fragments
            ) / weight
            rows[r1][c1] = combined
            confidences[r1][c1] = confidence

    @staticmethod
    def _is_cjk(character: str) -> bool:
        code = ord(character)
        return (
            0x3040 <= code <= 0x30FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xAC00 <= code <= 0xD7AF
            or 0x1100 <= code <= 0x11FF
            or 0xF900 <= code <= 0xFAFF
        )

    @staticmethod
    def _cell_crop(image_bgr, x: int, y: int, w: int, h: int):
        """Exclude only the rule itself, then add OCR-safe white context."""
        inset = max(1, min(3, min(w, h) // 12))
        x1, y1 = x + inset, y + inset
        x2, y2 = x + w - inset, y + h - inset
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0), dtype=np.uint8)
        crop = image_bgr[y1:y2, x1:x2]
        return cv2.copyMakeBorder(
            crop,
            4,
            4,
            4,
            4,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    @staticmethod
    def _has_ink(crop: np.ndarray) -> bool:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.size == 0:
            return False
        if min(gray.shape[:2]) > 8:
            gray = gray[3:-3, 3:-3]
        median = float(np.median(gray))
        threshold = min(210.0, median - 25.0)
        foreground = (gray < threshold).astype(np.uint8)
        if not np.any(foreground):
            return False

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8
        )
        meaningful_area = 0
        height, width = gray.shape[:2]
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_w = int(stats[label, cv2.CC_STAT_WIDTH])
            component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 3:
                continue
            # Residual table rules are long and only a few pixels thick.
            if component_w >= width * 0.82 and component_h <= 4:
                continue
            if component_h >= height * 0.82 and component_w <= 4:
                continue
            meaningful_area += area
        fraction = meaningful_area / float(max(gray.size, 1))
        return 0.0008 <= fraction <= 0.65

    @staticmethod
    def _has_content(table: ExtractedTable) -> bool:
        return table.row_count > 0 and table.col_count > 0 and any(
            (cell or "").strip() for row in table.rows for cell in row
        )

    def _ocr_full_page(self, spans: list[OcrSpan], source: Path) -> ExtractedTable:
        if not spans:
            return ExtractedTable(rows=[], source_image=source)

        items = sorted(spans, key=lambda span: (span.center[1], span.x1))
        heights = [span.height for span in items]
        row_threshold = max(float(sorted(heights)[len(heights) // 2]) * 0.7, 10.0)

        row_groups: list[list[OcrSpan]] = [[items[0]]]
        for entry in items[1:]:
            prev = row_groups[-1][-1]
            prev_cy = prev.center[1]
            cur_cy = entry.center[1]
            if abs(cur_cy - prev_cy) <= row_threshold:
                row_groups[-1].append(entry)
            else:
                row_groups.append([entry])

        rows: list[list[str]] = []
        confidences: list[list[float]] = []
        for group in row_groups:
            ordered = sorted(group, key=lambda span: span.x1)
            rows.append([span.text for span in ordered])
            confidences.append([span.confidence for span in ordered])

        return ExtractedTable(rows=rows, confidences=confidences, source_image=source)
