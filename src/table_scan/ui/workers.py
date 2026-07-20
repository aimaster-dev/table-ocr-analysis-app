"""Background workers so OCR never blocks the UI thread."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from table_scan.config.settings import AppSettings
from table_scan.models.table_result import ConversionResult
from table_scan.services.conversion_service import ConversionService


class ConversionWorker(QThread):
    """Runs batch conversion off the GUI thread."""

    progress = Signal(int, int, object)  # current, total, ConversionResult
    finished_ok = Signal(list)  # list[ConversionResult]
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        images: list[Path],
        output_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._images = images
        self._output_dir = output_dir
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            service = ConversionService(self._settings)
            engine = (self._settings.ocr_engine or "local").lower()
            if engine == "vlm":
                self.status.emit(
                    f"Checking local VLM ({self._settings.vlm_model}) via Ollama…"
                )
            elif engine == "paddle":
                self.status.emit(
                    f"Loading PaddleOCR ({self._settings.paddle_lang})… first run downloads models"
                )
            elif engine == "url":
                self.status.emit("Checking remote OCR endpoint…")
            else:
                self.status.emit("Checking Tesseract OCR…")
            service.warm_up()
            self.status.emit("OCR ready. Converting images…")

            def on_progress(current: int, total: int, result: ConversionResult) -> None:
                self.progress.emit(current, total, result)

            results = service.convert_many(
                self._images,
                self._output_dir,
                on_progress=on_progress,
                should_cancel=lambda: self._cancel_requested,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
