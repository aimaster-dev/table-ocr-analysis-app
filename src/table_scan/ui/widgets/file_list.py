"""Image list widget for the conversion queue."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from table_scan.models.table_result import JobStatus


class FileListWidget(QListWidget):
    """Shows queued image files and per-file status badges."""

    file_activated = Signal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.itemSelectionChanged.connect(self._emit_selection)

    def set_files(self, files: list[Path]) -> None:
        self.clear()
        for path in files:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole + 1, JobStatus.PENDING.value)
            self.addItem(item)

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
            label = f"{badge}  {path.name}"
            if detail:
                label = f"{label}  —  {detail}"
            item.setText(label)
            item.setData(Qt.ItemDataRole.UserRole + 1, status.value)
            break

    def paths(self) -> list[Path]:
        result: list[Path] = []
        for index in range(self.count()):
            item = self.item(index)
            result.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        return result

    def _emit_selection(self) -> None:
        items = self.selectedItems()
        if not items:
            return
        path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        self.file_activated.emit(path)
