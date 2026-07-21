"""Image list widget for the conversion queue."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QSizePolicy

from table_scan.models.table_result import JobStatus


class FileListWidget(QListWidget):
    """Shows queued image files and per-file status badges."""

    file_activated = Signal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setWordWrap(False)
        self.setUniformItemSizes(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.itemSelectionChanged.connect(self._emit_selection)

    def set_files(self, files: list[Path]) -> None:
        self.clear()
        for path in files:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setData(Qt.ItemDataRole.UserRole + 2, path.name)
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole + 1, JobStatus.PENDING.value)
            self.addItem(item)
        self._elide_visible_items()

    def update_status(self, path: Path, status: JobStatus, detail: str = "") -> None:
        target = str(path)
        for index in range(self.count()):
            item = self.item(index)
            if item.data(Qt.ItemDataRole.UserRole) != target:
                continue
            badge = {
                JobStatus.PENDING: "○",
                JobStatus.RUNNING: "…",
                JobStatus.SUCCESS: "✓",
                JobStatus.FAILED: "✗",
                JobStatus.SKIPPED: "–",
            }.get(status, "·")
            base = f"{badge}  {path.name}"
            full = f"{base}  —  {detail}" if detail else base
            item.setData(Qt.ItemDataRole.UserRole + 2, full)
            item.setToolTip(f"{path}\n{detail}" if detail else str(path))
            item.setData(Qt.ItemDataRole.UserRole + 1, status.value)
            item.setText(self._elided(full))
            break

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide_visible_items()

    def paths(self) -> list[Path]:
        result: list[Path] = []
        for index in range(self.count()):
            item = self.item(index)
            result.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        return result

    def _elide_visible_items(self) -> None:
        for index in range(self.count()):
            item = self.item(index)
            full = item.data(Qt.ItemDataRole.UserRole + 2) or item.text()
            item.setText(self._elided(str(full)))

    def _elided(self, text: str) -> str:
        # Keep a little padding so text never paints past the list border.
        width = max(40, self.viewport().width() - 24)
        return QFontMetrics(self.font()).elidedText(
            text, Qt.TextElideMode.ElideMiddle, width
        )

    def _emit_selection(self) -> None:
        items = self.selectedItems()
        if not items:
            return
        path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        self.file_activated.emit(path)
