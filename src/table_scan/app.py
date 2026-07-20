"""Qt application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from table_scan import __app_name__, __version__
from table_scan.config.settings import AppSettings
from table_scan.ui.main_window import MainWindow
from table_scan.utils.logging_config import setup_logging
from table_scan.utils.resource_path import resource_path


def run() -> int:
    setup_logging()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("TableScan")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    qss_path = resource_path("ui", "styles", "theme.qss")
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    settings = AppSettings.load()
    window = MainWindow(settings)
    window.show()

    return app.exec()
