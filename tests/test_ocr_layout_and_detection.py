"""Regression tests for mixed-script OCR layout and table geometry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import json

import cv2
import numpy as np
import pytest

from table_scan.config.settings import (
    OUTPUT_FORMAT_BOTH,
    OUTPUT_FORMAT_EXCEL,
    AppSettings,
)
from table_scan.core.layout_mapper import OcrSpan, compose_cell, map_spans_to_grid
from table_scan.core.ocr_engine import OcrEngine
from table_scan.core.preprocessor import ImagePreprocessor
from table_scan.core.paddle_ocr_engine import (
    MIXED_HANDWRITING_LANG,
    MIXED_HANDWRITING_MODEL,
    PaddleOcrEngine,
)
from table_scan.core.pipeline import TableExtractionPipeline
from table_scan.core.table_detector import TableDetector, TableGrid
from table_scan.models.table_result import JobStatus
from table_scan.models.table_result import ExtractedTable


def _box(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_page_spans_map_once_and_preserve_mixed_scripts() -> None:
    grid = TableGrid(
        cells=[
            [(0, 0, 100, 40), (100, 0, 100, 40)],
            [(0, 40, 100, 40), (100, 40, 100, 40)],
        ]
    )
    spans = [
        OcrSpan(10, 8, 45, 28, "商品", 0.96),
        OcrSpan(110, 8, 155, 28, "Price", 0.94),
        OcrSpan(10, 48, 52, 69, "한국어", 0.93),
        OcrSpan(115, 48, 160, 69, "１２３.45", 0.98),
    ]

    rows, confidences = map_spans_to_grid(spans, grid)

    assert rows == [["商品", "Price"], ["한국어", "１２３.45"]]
    assert confidences[1][1] == 0.98


def test_unicode_cleanup_is_domain_neutral() -> None:
    # Full-width forms normalize, but IDs, leading zeroes, commas, and ranges
    # must not be guessed or rewritten from a presumed price-table schema.
    assert OcrEngine.normalize_cell_value("ＡＢＣ－００７", col_index=0, col_count=5) == "ABC-007"
    assert OcrEngine.normalize_cell_value("0012", col_index=0, col_count=5) == "0012"
    assert OcrEngine.normalize_cell_value("50,00", col_index=2, col_count=5) == "50,00"
    assert OcrEngine.normalize_cell_value("型号 12-34", col_index=2, col_count=5) == "型号 12-34"
    assert OcrEngine.normalize_cell_value("備考", col_index=4, col_count=5) == "備考"
    assert OcrEngine.normalize_cell_value(
        "• First item\n• Second item", col_index=1, col_count=3
    ) == "• First item\n• Second item"
    assert OcrEngine.normalize_cell_value(
        "+ First + Second * Third +Fourth", col_index=1, col_count=3
    ) == "• First\n• Second\n• Third\n• Fourth"


def test_compose_cell_keeps_spaces_between_close_latin_words() -> None:
    text, _ = compose_cell(
        [
            OcrSpan(0, 0, 20, 10, "that", 0.95),
            OcrSpan(20.5, 0, 28, 10, "is", 0.95),
            OcrSpan(28.5, 0, 40, 10, "good", 0.95),
            OcrSpan(40, 0, 42, 10, ".", 0.95),
        ]
    )

    assert text == "that is good."


def test_paddle_2x_and_3x_results_normalize_without_shape_confusion() -> None:
    classic = [
        [
            [_box(0, 0, 30, 12), ("中文 A1", 0.97)],
            [_box(40, 0, 70, 12), ("123", 0.95)],
        ]
    ]
    assert [item[1] for item in PaddleOcrEngine._normalize_result(classic)] == [
        "中文 A1",
        "123",
    ]

    modern = [
        {
            "res": {
                "rec_texts": ["日本語", "42"],
                "rec_scores": np.asarray([0.91, 0.99]),
                "rec_boxes": np.asarray([[0, 0, 35, 14], [40, 0, 55, 14]]),
            }
        }
    ]
    normalized = PaddleOcrEngine._normalize_result(modern)
    assert [item[1] for item in normalized] == ["日本語", "42"]
    assert normalized[1][2] == 0.99


def test_mixed_handwriting_mode_uses_unified_ppocr_v6_model(monkeypatch) -> None:
    captured: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePipeline))
    engine = PaddleOcrEngine(paddle_lang=MIXED_HANDWRITING_LANG)

    engine.warm_up()

    assert captured["text_recognition_model_name"] == MIXED_HANDWRITING_MODEL
    assert "lang" not in captured
    assert captured["use_doc_unwarping"] is False
    assert captured["use_textline_orientation"] is True
    assert captured["enable_mkldnn"] is False


def test_mixed_page_upscale_restores_original_box_coordinates() -> None:
    class FakePipeline:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "rec_texts": ["한국 中文 A-12"],
                        "rec_scores": [0.96],
                        "rec_boxes": [[30, 60, 330, 180]],
                    }
                }
            ]

    engine = PaddleOcrEngine(paddle_lang=MIXED_HANDWRITING_LANG)
    engine._ocr = FakePipeline()
    image = np.full((500, 600, 3), 255, dtype=np.uint8)

    result = engine.read_text(image)

    assert result[0][0] == [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]
    assert result[0][1] == "한국 中文 A-12"


def test_onednn_runtime_failure_reports_the_stable_windows_fix() -> None:
    class BrokenPipeline:
        def predict(self, _image):
            raise NotImplementedError(
                "ConvertPirAttribute2RuntimeAttribute not support "
                "[pir::ArrayAttribute<pir::DoubleAttribute>] "
                "at onednn_instruction.cc:118"
            )

    engine = PaddleOcrEngine(paddle_lang=MIXED_HANDWRITING_LANG)
    engine._ocr = BrokenPipeline()

    with pytest.raises(RuntimeError, match="paddlepaddle==3.2.2"):
        engine.read_text(np.full((100, 100, 3), 255, dtype=np.uint8))


def test_mixed_handwriting_retries_weak_cell_with_binary_variant() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, _image):
            self.calls += 1
            text, score = ("한? 12", 0.51) if self.calls == 1 else ("한국 中文 12", 0.91)
            return [
                {
                    "res": {
                        "rec_texts": [text],
                        "rec_scores": [score],
                        "rec_boxes": [[5, 5, 180, 70]],
                    }
                }
            ]

    fake = FakePipeline()
    engine = PaddleOcrEngine(paddle_lang=MIXED_HANDWRITING_LANG)
    engine._ocr = fake
    cell = np.full((34, 140, 3), 240, dtype=np.uint8)
    cv2.putText(cell, "A12", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1)

    text, confidence = engine.read_cell(cell)

    assert fake.calls == 2
    assert text == "한국 中文 12"
    assert confidence == 0.91


def test_old_default_paddle_setting_migrates_to_mixed_handwriting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps({"ocr_engine": "paddle", "paddle_lang": "ch"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        AppSettings,
        "_config_path",
        classmethod(lambda cls: config),
    )

    settings = AppSettings.load()

    assert settings.settings_version == 3
    assert settings.paddle_lang == MIXED_HANDWRITING_LANG
    assert settings.output_format == OUTPUT_FORMAT_EXCEL


def test_new_settings_default_to_both_output_formats() -> None:
    assert AppSettings().output_format == OUTPUT_FORMAT_BOTH


def test_detector_separates_two_local_tables_from_page_frame() -> None:
    image = np.full((420, 720, 3), 255, dtype=np.uint8)
    # A page frame must not become an extra first/last table column.
    cv2.rectangle(image, (8, 8), (710, 410), (0, 0, 0), 2)
    _draw_grid(image, 40, 50, [90, 130, 80], [42, 42, 42])
    _draw_grid(image, 420, 170, [80, 100, 80], [38, 38, 38, 38])

    grids = TableDetector().detect(image)

    assert [(grid.rows, grid.cols) for grid in grids] == [(3, 3), (4, 3)]
    assert all(not grid.is_fallback for grid in grids)


def test_detector_preserves_a_merged_header_colspan() -> None:
    image = np.full((220, 360, 3), 255, dtype=np.uint8)
    x_values = [20, 120, 220, 320]
    y_values = [20, 70, 120, 170]
    for y in y_values:
        cv2.line(image, (20, y), (320, y), (0, 0, 0), 2)
    for x in (20, 320):
        cv2.line(image, (x, 20), (x, 170), (0, 0, 0), 2)
    for x in (120, 220):
        cv2.line(image, (x, 70), (x, 170), (0, 0, 0), 2)

    grid = TableDetector().detect(image)[0]

    assert (grid.rows, grid.cols) == (3, 3)
    assert grid.merged_ranges == [(0, 0, 0, 2)]


def test_detector_recovers_shaky_hand_drawn_grid() -> None:
    image = np.full((330, 500, 3), 255, dtype=np.uint8)
    x_values = [40, 170, 315, 455]
    y_values = [35, 110, 190, 275]
    for row_index, y in enumerate(y_values):
        values = list(range(40, 456, 25)) + [455]
        points = np.asarray(
            [
                [x, y + ((index + row_index) % 3 - 1) * 2]
                for index, x in enumerate(values)
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, (0, 0, 0), 2)
    for col_index, x in enumerate(x_values):
        values = list(range(35, 276, 22)) + [275]
        points = np.asarray(
            [
                [x + ((index + col_index) % 3 - 1) * 2, y]
                for index, y in enumerate(values)
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, (0, 0, 0), 2)

    grid = TableDetector().detect(image)[0]

    assert (grid.rows, grid.cols) == (3, 3)
    assert not grid.is_fallback


def test_detector_recovers_edge_clipped_example_without_hough_text_rows() -> None:
    image_path = Path(__file__).resolve().parents[1] / "example" / "1.webp"
    image = cv2.imread(str(image_path))
    assert image is not None
    processed = ImagePreprocessor().process(image)
    detector = TableDetector()

    grids = detector.detect(processed)

    assert len(grids) == 1
    grid = grids[0]
    assert grid.cols == 3
    assert grid.cells[0][0][0] > 0  # spreadsheet row-number gutter is excluded
    assert abs(grid.cells[0][-1][0] + grid.cells[0][-1][2] - (image.shape[1] - 1)) <= 1
    assert detector._used_hough is False


def test_pipeline_trims_empty_grid_bands_and_spreadsheet_column_chrome() -> None:
    grid = TableGrid(
        cells=[
            [(30, 0, 100, 20), (130, 0, 100, 20), (230, 0, 100, 20)],
            [(30, 20, 100, 30), (130, 20, 100, 30), (230, 20, 100, 30)],
            [(30, 50, 100, 50), (130, 50, 100, 50), (230, 50, 100, 50)],
            [(30, 100, 100, 20), (130, 100, 100, 20), (230, 100, 100, 20)],
        ]
    )
    table = ExtractedTable(
        rows=[
            ["A", "B", "C"],
            ["Topic", "Key Points", "Notes"],
            ["Time", "Use a calendar", "Work in the morning"],
            ["", "", ""],
        ],
        confidences=[[0.99, 0.99, 0.99]] * 4,
    )

    trimmed = TableExtractionPipeline._trim_table_edges(table, grid=grid)

    assert trimmed.rows == [
        ["Topic", "Key Points", "Notes"],
        ["Time", "Use a calendar", "Work in the morning"],
    ]
    assert len(trimmed.confidences) == 2


def test_focused_retry_keeps_page_detected_wrapping() -> None:
    current = "Want work thatis\nmeaningful + flexible.\nImpact > income.\nKeep learning always."

    value = TableExtractionPipeline._preserve_line_layout(
        "Want work that is meaningful + flexible. Impact > income. Keep learning always.",
        current,
    )

    assert "that is" in value
    assert value.count("\n") >= 2


def test_perspective_correction_recovers_photographed_grid() -> None:
    source = np.full((260, 380, 3), 255, dtype=np.uint8)
    for x in (10, 130, 250, 370):
        cv2.line(source, (x, 10), (x, 250), (0, 0, 0), 3)
    for y in (10, 90, 170, 250):
        cv2.line(source, (10, y), (370, y), (0, 0, 0), 3)
    source_points = np.float32([[0, 0], [379, 0], [379, 259], [0, 259]])
    photo_points = np.float32([[90, 70], [500, 35], [545, 360], [55, 390]])
    matrix = cv2.getPerspectiveTransform(source_points, photo_points)
    photo = cv2.warpPerspective(
        source,
        matrix,
        (600, 440),
        borderValue=(210, 210, 210),
    )

    corrected = ImagePreprocessor(
        deskew=True,
        enhance_contrast=False,
        rectify_perspective=True,
    ).process(photo)
    grid = TableDetector().detect(corrected)[0]

    assert corrected.shape[:2] != photo.shape[:2]
    assert (grid.rows, grid.cols) == (3, 3)


def test_pipeline_uses_one_page_pass_when_grid_is_recognized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = np.full((130, 220, 3), 255, dtype=np.uint8)
    _draw_grid(image, 10, 10, [100, 100], [50, 50])
    for text, origin in (("A", (20, 40)), ("Q", (125, 40)), ("B", (20, 90)), ("12", (130, 90))):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    image_path = tmp_path / "mixed.png"
    assert cv2.imwrite(str(image_path), image)

    class FakeOcr:
        def __init__(self) -> None:
            self.page_calls = 0
            self.cell_calls = 0

        def warm_up(self) -> None:
            return None

        def read_text(self, _image):
            self.page_calls += 1
            return [
                (_box(20, 20, 60, 40), "名称", 0.95),
                (_box(125, 20, 175, 40), "Qty", 0.95),
                (_box(20, 70, 70, 90), "りんご", 0.95),
                (_box(130, 70, 160, 90), "12", 0.95),
            ]

        def read_cell(self, _image):
            self.cell_calls += 1
            return "", 0.0

    fake = FakeOcr()
    monkeypatch.setattr("table_scan.core.pipeline.create_ocr_engine", lambda _settings: fake)
    settings = AppSettings(deskew=False, enhance_contrast=False)
    pipeline = TableExtractionPipeline(settings)

    result = pipeline.convert_image(image_path, tmp_path / "out")

    assert result.status == JobStatus.SUCCESS
    assert fake.page_calls == 1
    assert fake.cell_calls == 0
    assert result.tables[0].rows == [["名称", "Qty"], ["りんご", "12"]]
    assert {path.suffix for path in result.output_paths} == {".xlsx", ".html"}
    assert all(path.is_file() for path in result.output_paths)
    assert "名称" in (tmp_path / "out" / "mixed.html").read_text(encoding="utf-8")


def _draw_grid(
    image: np.ndarray,
    left: int,
    top: int,
    widths: list[int],
    heights: list[int],
) -> None:
    x_values = [left]
    for width in widths:
        x_values.append(x_values[-1] + width)
    y_values = [top]
    for height in heights:
        y_values.append(y_values[-1] + height)
    for x in x_values:
        cv2.line(image, (x, y_values[0]), (x, y_values[-1]), (0, 0, 0), 2)
    for y in y_values:
        cv2.line(image, (x_values[0], y), (x_values[-1], y), (0, 0, 0), 2)
