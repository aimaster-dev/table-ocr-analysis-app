"""Map page-level OCR boxes into a detected table grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from table_scan.core.table_detector import TableGrid


@dataclass(frozen=True, slots=True)
class OcrSpan:
    """One recognized text span in page coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_result(cls, item: Any) -> OcrSpan | None:
        """Parse an EasyOCR/Paddle-compatible ``(box, text, conf)`` item."""
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return None
        box, text = item[0], str(item[1]).strip()
        if not text:
            return None
        try:
            points = np.asarray(box, dtype=float).reshape(-1, 2)
            confidence = float(item[2])
        except (TypeError, ValueError):
            return None
        if len(points) < 2 or not np.isfinite(points).all():
            return None
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        if x2 <= x1 or y2 <= y1:
            return None
        if confidence > 1.0:
            confidence /= 100.0
        confidence = min(1.0, max(0.0, confidence))
        return cls(float(x1), float(y1), float(x2), float(y2), text, confidence)


def parse_ocr_spans(results: list[Any]) -> list[OcrSpan]:
    """Discard malformed OCR output without discarding valid neighboring spans."""
    spans: list[OcrSpan] = []
    for item in results or []:
        span = OcrSpan.from_result(item)
        if span is not None:
            spans.append(span)
    return spans


def map_spans_to_grid(
    spans: list[OcrSpan],
    grid: TableGrid,
) -> tuple[list[list[str]], list[list[float]]]:
    """Assign every span to at most one cell, then restore reading order."""
    assignments: dict[tuple[int, int], list[OcrSpan]] = {}
    flat_cells: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for row_index, row in enumerate(grid.cells):
        for col_index, box in enumerate(row):
            flat_cells.append((row_index, col_index, box))

    for span in spans:
        best: tuple[int, int] | None = None
        best_score = 0.0
        cx, cy = span.center
        for row_index, col_index, (x, y, w, h) in flat_cells:
            # Center containment is stable even when a detector's quadrilateral
            # extends slightly over a table rule.
            if x <= cx <= x + w and y <= cy <= y + h:
                score = 2.0 + _intersection_ratio(span, (x, y, w, h))
            else:
                score = _intersection_ratio(span, (x, y, w, h))
            if score > best_score:
                best_score = score
                best = (row_index, col_index)
        if best is not None and best_score >= 0.45:
            assignments.setdefault(best, []).append(span)

    rows: list[list[str]] = []
    confidences: list[list[float]] = []
    for row_index, row in enumerate(grid.cells):
        texts: list[str] = []
        confs: list[float] = []
        for col_index, _ in enumerate(row):
            text, confidence = compose_cell(assignments.get((row_index, col_index), []))
            texts.append(text)
            confs.append(confidence)
        rows.append(texts)
        confidences.append(confs)
    return rows, confidences


def compose_cell(spans: list[OcrSpan]) -> tuple[str, float]:
    """Compose one or more OCR spans while respecting CJK word boundaries."""
    if not spans:
        return "", 0.0

    median_height = float(np.median([max(span.height, 1.0) for span in spans]))
    line_tolerance = max(3.0, median_height * 0.55)
    ordered = sorted(spans, key=lambda span: (span.center[1], span.x1))
    lines: list[list[OcrSpan]] = []
    centers: list[float] = []
    for span in ordered:
        cy = span.center[1]
        target = next(
            (index for index, center in enumerate(centers) if abs(cy - center) <= line_tolerance),
            None,
        )
        if target is None:
            lines.append([span])
            centers.append(cy)
        else:
            lines[target].append(span)
            centers[target] = sum(item.center[1] for item in lines[target]) / len(lines[target])

    line_order = sorted(range(len(lines)), key=lambda index: centers[index])
    text_lines: list[str] = []
    all_spans: list[OcrSpan] = []
    for index in line_order:
        line = sorted(lines[index], key=lambda span: span.x1)
        all_spans.extend(line)
        pieces = [line[0].text]
        previous = line[0]
        for current in line[1:]:
            gap = current.x1 - previous.x2
            close = gap <= max(2.0, min(previous.height, current.height) * 0.12)
            separator = "" if close or _cjk_boundary(previous.text, current.text) else " "
            pieces.append(separator + current.text)
            previous = current
        text_lines.append("".join(pieces).strip())

    weight = sum(max(len(span.text), 1) for span in all_spans)
    confidence = (
        sum(span.confidence * max(len(span.text), 1) for span in all_spans) / weight
        if weight
        else 0.0
    )
    return "\n".join(line for line in text_lines if line), confidence


def _intersection_ratio(span: OcrSpan, cell: tuple[int, int, int, int]) -> float:
    x, y, w, h = cell
    x1, y1 = max(span.x1, x), max(span.y1, y)
    x2, y2 = min(span.x2, x + w), min(span.y2, y + h)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / span.area if span.area > 0 else 0.0


def _cjk_boundary(left: str, right: str) -> bool:
    return bool(left and right and (_is_cjk(left[-1]) or _is_cjk(right[0])))


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana / Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
        or 0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
    )
