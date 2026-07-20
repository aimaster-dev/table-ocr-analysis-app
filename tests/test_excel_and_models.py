"""Smoke tests for pure helpers (no OCR / Qt required)."""

from __future__ import annotations

from table_scan.core.excel_writer import ExcelExporter
from table_scan.models.table_result import ExtractedTable
from openpyxl import load_workbook


def test_extracted_table_pads_ragged_rows() -> None:
    table = ExtractedTable(rows=[["a", "b"], ["c"], ["d", "e", "f"]])
    rect = table.to_rectangular()
    assert rect == [["a", "b", ""], ["c", "", ""], ["d", "e", "f"]]


def test_excel_exporter_writes_file(tmp_path) -> None:
    table = ExtractedTable(rows=[["Name", "Qty"], ["Apple", "3"], ["Pear", "1"]])
    out = tmp_path / "sample.xlsx"
    ExcelExporter().export([table], out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_excel_exporter_preserves_detected_merged_header(tmp_path) -> None:
    table = ExtractedTable(
        rows=[["販売報告", "", ""], ["商品", "数量", "Price"]],
        merged_ranges=[(0, 0, 0, 2)],
    )
    out = tmp_path / "merged.xlsx"
    ExcelExporter().export([table], out)

    sheet = load_workbook(out).active
    assert str(next(iter(sheet.merged_cells.ranges))) == "A1:C1"
    assert sheet["A1"].value == "販売報告"


def test_excel_exporter_flags_low_confidence_handwriting(tmp_path) -> None:
    table = ExtractedTable(
        rows=[["항목", "Value"], ["손글씨", "12"]],
        confidences=[[0.98, 0.97], [0.51, 0.94]],
    )
    out = tmp_path / "review.xlsx"
    ExcelExporter().export([table], out)

    sheet = load_workbook(out).active
    assert sheet["A2"].fill.fill_type == "solid"
    assert sheet["A2"].fill.fgColor.rgb == "00FFF2CC"
    assert sheet["B2"].fill.fill_type is None
