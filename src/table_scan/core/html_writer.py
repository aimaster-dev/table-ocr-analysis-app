"""Write extracted tables to secure, self-contained HTML documents."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path

from table_scan.models.table_result import ExtractedTable


_STYLES = """
:root {
  color-scheme: light;
  font-family: "Segoe UI", "Noto Sans", Arial, sans-serif;
  color: #102a43;
  background: #f4f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; background: #f4f6f8; }
.document { width: min(1500px, 100%); margin: 0 auto; padding: 24px; }
.document-header { margin-bottom: 18px; }
h1 { margin: 0 0 6px; font-size: 1.5rem; }
.summary { margin: 0; color: #627d98; }
.table-section {
  margin: 0 0 24px;
  padding: 18px;
  overflow: hidden;
  border: 1px solid #d9e2ec;
  border-radius: 10px;
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(16, 42, 67, 0.06);
}
.table-section h2 { margin: 0 0 12px; font-size: 1.05rem; }
.table-scroll { overflow: auto; }
table { width: 100%; min-width: 42rem; border-collapse: collapse; table-layout: auto; }
th, td {
  padding: 10px 12px;
  border: 1px solid #b0b8c0;
  vertical-align: top;
  text-align: left;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.35;
}
thead th { position: sticky; top: 0; z-index: 1; background: #243b53; color: #ffffff; }
tbody tr:nth-child(even) td { background: #f8fafc; }
th.review, td.review { background: #fff2cc; color: #5c4400; }
th.empty, td.empty { min-width: 4rem; }
@media print {
  :root, body { background: #ffffff; }
  .document { width: 100%; padding: 0; }
  .table-section { break-inside: avoid; border: 0; box-shadow: none; padding: 0; }
  .table-scroll { overflow: visible; }
  table { min-width: 0; }
  thead th { position: static; }
}
""".strip()


class HtmlExporter:
    """Create one offline HTML document containing all extracted tables."""

    def __init__(self, *, review_confidence: float = 0.65) -> None:
        self.review_confidence = review_confidence

    def export(
        self,
        tables: list[ExtractedTable],
        output_path: Path,
        *,
        title: str | None = None,
    ) -> Path:
        if not tables:
            raise ValueError("No tables to export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document_title = title or output_path.stem
        markup = self.render(tables, title=document_title)

        # Publish atomically so a cancelled or failed write never leaves a
        # partially valid HTML document that looks like a successful result.
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        try:
            temporary.write_text(markup, encoding="utf-8", newline="\n")
            temporary.replace(output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return output_path

    def render(self, tables: list[ExtractedTable], *, title: str) -> str:
        safe_title = escape(str(title), quote=True)
        parts = [
            "<!doctype html>",
            '<html lang="und">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta http-equiv="Content-Security-Policy" '
            'content="default-src \'none\'; style-src \'unsafe-inline\'">',
            f"<title>{safe_title}</title>",
            f"<style>{_STYLES}</style>",
            "</head>",
            "<body>",
            '<main class="document">',
            '<header class="document-header">',
            f"<h1>{safe_title}</h1>",
            f'<p class="summary">{len(tables)} extracted table(s)</p>',
            "</header>",
        ]
        for index, table in enumerate(tables, start=1):
            parts.extend(self._render_table(table, index=index))
        parts.extend(["</main>", "</body>", "</html>", ""])
        return "\n".join(parts)

    def _render_table(self, table: ExtractedTable, *, index: int) -> list[str]:
        matrix = table.to_rectangular()
        row_count = len(matrix)
        col_count = table.col_count
        anchors, covered = self._merge_layout(
            row_count,
            col_count,
            table.merged_ranges,
        )
        parts = [
            f'<section class="table-section" data-table-index="{index}">',
            f"<h2>Table {index}</h2>",
            '<div class="table-scroll">',
            f'<table aria-label="Extracted table {index}">',
        ]
        if col_count:
            parts.append("<colgroup>")
            for width in self._column_widths(matrix, col_count):
                parts.append(f'<col style="width:{width}ch">')
            parts.append("</colgroup>")

        if row_count:
            parts.append("<thead>")
            parts.extend(
                self._render_row(
                    table,
                    matrix,
                    row=0,
                    header=True,
                    anchors=anchors,
                    covered=covered,
                )
            )
            parts.append("</thead>")
        parts.append("<tbody>")
        for row in range(1, row_count):
            parts.extend(
                self._render_row(
                    table,
                    matrix,
                    row=row,
                    header=False,
                    anchors=anchors,
                    covered=covered,
                )
            )
        parts.extend(["</tbody>", "</table>", "</div>", "</section>"])
        return parts

    def _render_row(
        self,
        table: ExtractedTable,
        matrix: list[list[str]],
        *,
        row: int,
        header: bool,
        anchors: dict[tuple[int, int], tuple[int, int]],
        covered: set[tuple[int, int]],
    ) -> list[str]:
        parts = ["<tr>"]
        tag = "th" if header else "td"
        for column, raw_value in enumerate(matrix[row]):
            coordinate = (row, column)
            if coordinate in covered:
                continue
            value = str(raw_value or "")
            confidence = self._confidence_at(table, row, column)
            classes: list[str] = []
            if not value.strip():
                classes.append("empty")
            if 0.0 < confidence < self.review_confidence:
                classes.append("review")
            attributes = []
            if header:
                attributes.append('scope="col"')
            if classes:
                class_names = " ".join(classes)
                attributes.append(f'class="{class_names}"')
            if confidence > 0.0:
                attributes.append(f'data-confidence="{confidence:.4f}"')
                attributes.append(f'title="OCR confidence: {confidence:.1%}"')
            rowspan, colspan = anchors.get(coordinate, (1, 1))
            if rowspan > 1:
                attributes.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attributes.append(f'colspan="{colspan}"')
            attr_text = " " + " ".join(attributes) if attributes else ""
            parts.append(f"<{tag}{attr_text}>{escape(value, quote=False)}</{tag}>")
        parts.append("</tr>")
        return parts

    @staticmethod
    def _confidence_at(table: ExtractedTable, row: int, column: int) -> float:
        try:
            value = float(table.confidences[row][column])
        except (IndexError, TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0

    @staticmethod
    def _column_widths(matrix: list[list[str]], col_count: int) -> list[int]:
        widths: list[int] = []
        for column in range(col_count):
            longest = 0
            for row in matrix:
                value = str(row[column] if column < len(row) else "")
                longest = max(longest, *(len(line) for line in (value.splitlines() or [""])))
            widths.append(min(48, max(12, longest + 2)))
        return widths

    @staticmethod
    def _merge_layout(
        row_count: int,
        col_count: int,
        merged_ranges: list[tuple[int, int, int, int]],
    ) -> tuple[dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]]]:
        anchors: dict[tuple[int, int], tuple[int, int]] = {}
        occupied: set[tuple[int, int]] = set()
        covered: set[tuple[int, int]] = set()
        for raw_range in sorted(merged_ranges):
            if len(raw_range) != 4 or row_count == 0 or col_count == 0:
                continue
            r1, c1, r2, c2 = raw_range
            r1, r2 = max(0, r1), min(row_count - 1, r2)
            c1, c2 = max(0, c1), min(col_count - 1, c2)
            if r2 < r1 or c2 < c1 or (r1 == r2 and c1 == c2):
                continue
            cells = {
                (row, column)
                for row in range(r1, r2 + 1)
                for column in range(c1, c2 + 1)
            }
            if cells & occupied:
                continue
            anchor = (r1, c1)
            anchors[anchor] = (r2 - r1 + 1, c2 - c1 + 1)
            occupied.update(cells)
            covered.update(cells - {anchor})
        return anchors, covered
