"""Primary application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from table_scan import __app_name__, __version__
from table_scan.config.settings import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_PADDLE_LANG,
    DEFAULT_VLM_MODEL,
    OCR_ENGINE_LOCAL,
    OCR_ENGINE_PADDLE,
    OCR_ENGINE_URL,
    OCR_ENGINE_VLM,
    OUTPUT_FORMAT_BOTH,
    OUTPUT_FORMAT_EXCEL,
    OUTPUT_FORMAT_HTML,
    AppSettings,
)
from table_scan.core.paddle_ocr_engine import PADDLE_LANG_CHOICES
from table_scan.models.table_result import ConversionResult, JobStatus
from table_scan.services.conversion_service import ConversionService
from table_scan.ui.widgets.file_list import FileListWidget
from table_scan.ui.widgets.image_preview import ImagePreview
from table_scan.ui.workers import ConversionWorker

logger = logging.getLogger(__name__)


class _SidebarScrollArea(QScrollArea):
    """Scroll area that locks content width to the viewport (no horizontal bleed)."""

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        child = self.widget()
        if child is not None:
            child.setFixedWidth(self.viewport().width())


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._worker: ConversionWorker | None = None
        self._images: list[Path] = []

        self.setWindowTitle(f"{__app_name__} {__version__}")
        self.resize(1180, 740)
        self.setMinimumSize(960, 600)

        self._build_actions()
        self._build_ui()
        self._build_menu()
        self._restore_paths()
        self._set_busy(False)

    # ------------------------------------------------------------------ UI
    def _build_actions(self) -> None:
        self.act_open_dir = QAction("Open Image Folder…", self)
        self.act_open_dir.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open_dir.triggered.connect(self._browse_input)

        self.act_convert = QAction("Convert", self)
        self.act_convert.setShortcut(QKeySequence("Ctrl+Return"))
        self.act_convert.triggered.connect(self._start_conversion)

        self.act_cancel = QAction("Cancel", self)
        self.act_cancel.setShortcut(QKeySequence("Esc"))
        self.act_cancel.triggered.connect(self._cancel_conversion)

        self.act_quit = QAction("Exit", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_open_dir)
        file_menu.addSeparator()
        file_menu.addAction(self.act_convert)
        file_menu.addAction(self.act_cancel)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("CentralRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(12)

        root_layout.addLayout(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_main_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setSizes([340, 840])
        root_layout.addWidget(splitter, stretch=1)

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label, stretch=1)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel(__app_name__)
        title.setObjectName("AppTitle")
        subtitle = QLabel("Convert photographed paper tables into Excel and HTML")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addStretch(1)
        return layout

    def _build_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setMinimumWidth(300)
        frame.setMaximumWidth(400)
        frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = _SidebarScrollArea()
        scroll.setObjectName("SidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        form = QWidget()
        form.setObjectName("SidebarForm")
        layout = QVBoxLayout(form)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        def constrain(widget: QWidget) -> QWidget:
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            return widget

        def add_combo(combo: QComboBox) -> None:
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
            constrain(combo)
            layout.addWidget(combo)

        layout.addWidget(self._section("Input folder"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select a folder of table photos…")
        self.input_edit.setReadOnly(True)
        constrain(self.input_edit)
        browse_in = QToolButton()
        browse_in.setText("Browse…")
        browse_in.clicked.connect(self._browse_input)
        row_in = QHBoxLayout()
        row_in.setSpacing(6)
        row_in.addWidget(self.input_edit, stretch=1)
        row_in.addWidget(browse_in)
        layout.addLayout(row_in)

        layout.addWidget(self._section("Output folder"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Where converted files will be saved…")
        constrain(self.output_edit)
        browse_out = QToolButton()
        browse_out.setText("Browse…")
        browse_out.clicked.connect(self._browse_output)
        row_out = QHBoxLayout()
        row_out.setSpacing(6)
        row_out.addWidget(self.output_edit, stretch=1)
        row_out.addWidget(browse_out)
        layout.addLayout(row_out)

        layout.addWidget(self._section("Output format"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("Excel workbook (.xlsx)", OUTPUT_FORMAT_EXCEL)
        self.output_format_combo.addItem("HTML document (.html)", OUTPUT_FORMAT_HTML)
        self.output_format_combo.addItem("Excel + HTML", OUTPUT_FORMAT_BOTH)
        format_index = self.output_format_combo.findData(self.settings.output_format)
        self.output_format_combo.setCurrentIndex(format_index if format_index >= 0 else 2)
        self.output_format_combo.currentIndexChanged.connect(
            self._on_output_format_changed
        )
        add_combo(self.output_format_combo)

        layout.addWidget(self._section("Options"))
        self.chk_deskew = QCheckBox("Auto-deskew photos")
        self.chk_deskew.setChecked(self.settings.deskew)
        self.chk_perspective = QCheckBox("Correct photographed perspective")
        self.chk_perspective.setChecked(self.settings.rectify_perspective)
        self.chk_contrast = QCheckBox("Enhance contrast")
        self.chk_contrast.setChecked(self.settings.enhance_contrast)
        layout.addWidget(self.chk_deskew)
        layout.addWidget(self.chk_perspective)
        layout.addWidget(self.chk_contrast)

        layout.addWidget(self._section("OCR engine"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Local Tesseract (printed)", OCR_ENGINE_LOCAL)
        self.engine_combo.addItem(
            "Local PaddleOCR (East Asian / handwriting)", OCR_ENGINE_PADDLE
        )
        self.engine_combo.addItem("Local VLM / Ollama (handwriting)", OCR_ENGINE_VLM)
        self.engine_combo.addItem("Remote URL (HTTP API)", OCR_ENGINE_URL)
        engine_index = self.engine_combo.findData(self.settings.ocr_engine)
        self.engine_combo.setCurrentIndex(engine_index if engine_index >= 0 else 0)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        add_combo(self.engine_combo)

        self.ocr_location_label = QLabel("Engine directory / URL")
        self.ocr_location_label.setObjectName("SectionLabel")
        layout.addWidget(self.ocr_location_label)
        self.ocr_location_edit = QLineEdit()
        self.ocr_location_edit.setText(
            self.settings.ocr_location or self.settings.tesseract_cmd
        )
        constrain(self.ocr_location_edit)
        self.browse_ocr_btn = QToolButton()
        self.browse_ocr_btn.setText("Browse…")
        self.browse_ocr_btn.clicked.connect(self._browse_ocr_location)
        row_ocr = QHBoxLayout()
        row_ocr.setSpacing(6)
        row_ocr.addWidget(self.ocr_location_edit, stretch=1)
        row_ocr.addWidget(self.browse_ocr_btn)
        layout.addLayout(row_ocr)

        self.paddle_lang_label = QLabel("PaddleOCR language")
        self.paddle_lang_label.setObjectName("SectionLabel")
        layout.addWidget(self.paddle_lang_label)
        self.paddle_lang_combo = QComboBox()
        for label, code in PADDLE_LANG_CHOICES:
            self.paddle_lang_combo.addItem(label, code)
        paddle_idx = self.paddle_lang_combo.findData(
            self.settings.paddle_lang or DEFAULT_PADDLE_LANG
        )
        self.paddle_lang_combo.setCurrentIndex(paddle_idx if paddle_idx >= 0 else 0)
        add_combo(self.paddle_lang_combo)

        self.paddle_model_label = QLabel("Paddle models directory (offline)")
        self.paddle_model_label.setObjectName("SectionLabel")
        layout.addWidget(self.paddle_model_label)
        self.paddle_model_edit = QLineEdit()
        self.paddle_model_edit.setText(self.settings.paddle_model_dir or "")
        self.paddle_model_edit.setPlaceholderText(
            str(Path.home() / ".paddlex" / "official_models")
        )
        constrain(self.paddle_model_edit)
        self.browse_paddle_btn = QToolButton()
        self.browse_paddle_btn.setText("Browse…")
        self.browse_paddle_btn.clicked.connect(self._browse_paddle_models)
        row_paddle = QHBoxLayout()
        row_paddle.setSpacing(6)
        row_paddle.addWidget(self.paddle_model_edit, stretch=1)
        row_paddle.addWidget(self.browse_paddle_btn)
        layout.addLayout(row_paddle)

        self.paddle_lang_hint = QLabel(
            "Leave empty to auto-download once, or choose extracted model folders."
        )
        self.paddle_lang_hint.setWordWrap(True)
        self.paddle_lang_hint.setObjectName("AppSubtitle")
        layout.addWidget(self.paddle_lang_hint)

        self.vlm_model_label = QLabel("VLM model")
        self.vlm_model_label.setObjectName("SectionLabel")
        layout.addWidget(self.vlm_model_label)
        self.vlm_model_edit = QLineEdit()
        self.vlm_model_edit.setText(self.settings.vlm_model or DEFAULT_VLM_MODEL)
        self.vlm_model_edit.setPlaceholderText(DEFAULT_VLM_MODEL)
        constrain(self.vlm_model_edit)
        layout.addWidget(self.vlm_model_edit)
        self._on_engine_changed()

        layout.addWidget(self._section("Images"))
        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(100)
        self.file_list.setMaximumHeight(160)
        layout.addWidget(self.file_list)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        constrain(self.progress)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.setObjectName("PrimaryButton")
        self.btn_convert.setToolTip("Start conversion with the selected output format")
        self.btn_convert.clicked.connect(self._start_conversion)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.clicked.connect(self._cancel_conversion)
        self.btn_open_out = QPushButton("Output")
        self.btn_open_out.setToolTip("Open the output folder")
        self.btn_open_out.clicked.connect(self._open_output_folder)
        for btn in (self.btn_convert, self.btn_cancel, self.btn_open_out):
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setMinimumWidth(0)
            btn_row.addWidget(btn, stretch=1)
        layout.addLayout(btn_row)

        scroll.setWidget(form)
        outer.addWidget(scroll, stretch=1)
        self._on_output_format_changed()
        return frame

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        preview_card = QFrame()
        preview_card.setObjectName("PreviewCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.addWidget(self._section("Preview"))
        self.preview = ImagePreview()
        preview_layout.addWidget(self.preview, stretch=1)
        layout.addWidget(preview_card, stretch=3)

        log_card = QFrame()
        log_card.setObjectName("LogCard")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self._section("Activity"))
        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_card, stretch=1)
        return panel

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionLabel")
        return label

    # -------------------------------------------------------------- helpers
    def _restore_paths(self) -> None:
        if self.settings.last_input_dir:
            self.input_edit.setText(self.settings.last_input_dir)
            self._load_directory(Path(self.settings.last_input_dir), silent=True)
        if self.settings.last_output_dir:
            self.output_edit.setText(self.settings.last_output_dir)

        # _restore_paths runs once during construction, so there is no previous
        # connection to remove.  Calling parameterless disconnect on a fresh
        # PySide signal emits a noisy libpyside RuntimeWarning on Windows.
        self.file_list.file_activated.connect(self.preview.show_image)

    def _browse_input(self) -> None:
        start = self.input_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if not chosen:
            return
        self.input_edit.setText(chosen)
        self.settings.last_input_dir = chosen
        self.settings.save()
        self._load_directory(Path(chosen))

    def _browse_output(self) -> None:
        start = self.output_edit.text() or self.input_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if not chosen:
            return
        self.output_edit.setText(chosen)
        self.settings.last_output_dir = chosen
        self.settings.save()

    def _load_directory(self, directory: Path, *, silent: bool = False) -> None:
        try:
            images = ConversionService.discover_images(directory)
        except (NotADirectoryError, OSError) as exc:
            if not silent:
                QMessageBox.warning(self, "Folder error", str(exc))
            return

        self._images = images
        self.file_list.set_files(images)
        self.preview.clear()
        self.progress.setValue(0)
        msg = f"Found {len(images)} image(s) in {directory}"
        self._log(msg)
        self.status_label.setText(msg)
        if images:
            self.file_list.setCurrentRow(0)
            self.preview.show_image(images[0])
        elif not silent:
            QMessageBox.information(
                self,
                "No images",
                "No supported images found.\n\nSupported: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP",
            )

    def _on_engine_changed(self) -> None:
        engine = self.engine_combo.currentData()
        is_tesseract = engine == OCR_ENGINE_LOCAL
        is_paddle = engine == OCR_ENGINE_PADDLE
        is_vlm = engine == OCR_ENGINE_VLM

        self.ocr_location_label.setVisible(not is_paddle)
        self.ocr_location_edit.setVisible(not is_paddle)
        self.browse_ocr_btn.setVisible(is_tesseract)
        self.paddle_lang_label.setVisible(is_paddle)
        self.paddle_lang_combo.setVisible(is_paddle)
        self.paddle_model_label.setVisible(is_paddle)
        self.paddle_model_edit.setVisible(is_paddle)
        self.browse_paddle_btn.setVisible(is_paddle)
        self.paddle_lang_hint.setVisible(is_paddle)
        self.vlm_model_label.setVisible(is_vlm)
        self.vlm_model_edit.setVisible(is_vlm)

        if is_tesseract:
            self.ocr_location_label.setText("Engine directory")
            self.ocr_location_edit.setPlaceholderText(r"C:\Program Files\Tesseract-OCR")
            current = self.ocr_location_edit.text().strip()
            if not current or current.lower().startswith("http"):
                self.ocr_location_edit.setText(r"C:\Program Files\Tesseract-OCR")
        elif is_vlm:
            self.ocr_location_label.setText("Ollama URL (offline / localhost)")
            self.ocr_location_edit.setPlaceholderText(DEFAULT_OLLAMA_URL)
            current = self.ocr_location_edit.text().strip()
            if not current or not current.lower().startswith("http"):
                self.ocr_location_edit.setText(DEFAULT_OLLAMA_URL)
        elif engine == OCR_ENGINE_URL:
            self.ocr_location_label.setText("Engine URL")
            self.ocr_location_edit.setPlaceholderText("https://ocr.example.com/v1/recognize")

    def _on_output_format_changed(self) -> None:
        output_format = self.output_format_combo.currentData()
        tips = {
            OUTPUT_FORMAT_EXCEL: "Convert to Excel workbook (.xlsx)",
            OUTPUT_FORMAT_HTML: "Convert to HTML document (.html)",
            OUTPUT_FORMAT_BOTH: "Convert to Excel + HTML",
        }
        tip = tips.get(output_format, "Convert")
        self.act_convert.setText("Convert")
        self.act_convert.setToolTip(tip)
        if hasattr(self, "btn_convert"):
            self.btn_convert.setText("Convert")
            self.btn_convert.setToolTip(tip)

    def _browse_ocr_location(self) -> None:
        start = self.ocr_location_edit.text().strip() or r"C:\Program Files\Tesseract-OCR"
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Tesseract-OCR directory",
            start,
        )
        if not chosen:
            return
        self.ocr_location_edit.setText(chosen)
        self.settings.ocr_engine = OCR_ENGINE_LOCAL
        self.settings.ocr_location = chosen
        self.settings.save()

    def _browse_paddle_models(self) -> None:
        start = self.paddle_model_edit.text().strip() or str(
            Path.home() / ".paddlex" / "official_models"
        )
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Paddle models directory",
            start,
        )
        if not chosen:
            return
        self.paddle_model_edit.setText(chosen)
        self.settings.ocr_engine = OCR_ENGINE_PADDLE
        self.settings.paddle_model_dir = chosen
        self.settings.save()

    def _persist_options(self) -> None:
        self.settings.deskew = self.chk_deskew.isChecked()
        self.settings.rectify_perspective = self.chk_perspective.isChecked()
        self.settings.enhance_contrast = self.chk_contrast.isChecked()
        self.settings.output_format = str(
            self.output_format_combo.currentData() or OUTPUT_FORMAT_BOTH
        )
        self.settings.ocr_engine = str(self.engine_combo.currentData())
        self.settings.ocr_location = self.ocr_location_edit.text().strip()
        self.settings.vlm_model = self.vlm_model_edit.text().strip() or DEFAULT_VLM_MODEL
        self.settings.paddle_lang = str(
            self.paddle_lang_combo.currentData() or DEFAULT_PADDLE_LANG
        )
        self.settings.paddle_model_dir = self.paddle_model_edit.text().strip()
        self.settings.last_input_dir = self.input_edit.text().strip()
        self.settings.last_output_dir = self.output_edit.text().strip()
        self.settings.save()

    def _start_conversion(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        if not self._images:
            QMessageBox.information(self, "Nothing to convert", "Choose a folder that contains table photos.")
            return

        output = self.output_edit.text().strip()
        if not output:
            # Default beside input.
            output = str(Path(self.input_edit.text()) / "table_output")
            self.output_edit.setText(output)

        self._persist_options()
        self._log(f"Starting conversion of {len(self._images)} image(s)…")
        self.progress.setRange(0, len(self._images))
        self.progress.setValue(0)
        self._set_busy(True)

        self._worker = ConversionWorker(
            self.settings,
            list(self._images),
            Path(output),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_conversion(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._log("Cancel requested… finishing current image.")
            self.status_label.setText("Cancelling…")

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)
        self._log(message)

    def _on_progress(self, current: int, total: int, result: object) -> None:
        assert isinstance(result, ConversionResult)
        self.progress.setMaximum(total)
        self.progress.setValue(current)

        detail = result.message
        if result.status == JobStatus.SUCCESS and result.output_paths:
            detail = ", ".join(path.name for path in result.output_paths)
        elif result.status == JobStatus.SUCCESS and result.output_path:
            detail = result.output_path.name
        self.file_list.update_status(result.image_path, result.status, detail)

        icon = "OK" if result.status == JobStatus.SUCCESS else result.status.value.upper()
        self._log(f"[{current}/{total}] {icon}: {result.image_path.name} — {result.message}")
        self.status_label.setText(f"Processed {current} of {total}")

    def _on_finished(self, results: list) -> None:
        self._set_busy(False)
        ok = sum(1 for r in results if r.status == JobStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == JobStatus.FAILED)
        skipped = sum(1 for r in results if r.status == JobStatus.SKIPPED)
        summary = f"Done. Success: {ok}  Failed: {failed}  Skipped: {skipped}"
        self._log(summary)
        self.status_label.setText(summary)
        QMessageBox.information(self, "Conversion complete", summary)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._log(f"ERROR: {message}")
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Conversion failed", message)

    def _set_busy(self, busy: bool) -> None:
        self.btn_convert.setEnabled(not busy)
        self.act_convert.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.act_cancel.setEnabled(busy)
        self.chk_deskew.setEnabled(not busy)
        self.chk_perspective.setEnabled(not busy)
        self.chk_contrast.setEnabled(not busy)
        self.output_format_combo.setEnabled(not busy)
        self.engine_combo.setEnabled(not busy)
        self.ocr_location_edit.setEnabled(not busy)
        self.browse_ocr_btn.setEnabled(not busy)
        self.vlm_model_edit.setEnabled(not busy)
        self.paddle_lang_combo.setEnabled(not busy)
        self.paddle_model_edit.setEnabled(not busy)
        self.browse_paddle_btn.setEnabled(not busy)

    def _open_output_folder(self) -> None:
        path = self.output_edit.text().strip()
        if not path:
            return
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(folder.as_uri())

    def _log(self, message: str) -> None:
        self.log_view.append(message)
        logger.info(message)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"<b>{__app_name__}</b> v{__version__}<br><br>"
            "Select a folder of table photos, then convert them into Excel (.xlsx), "
            "HTML (.html), or both.<br><br>"
            "OCR: Tesseract, PaddleOCR, or offline VLM · UI: Qt (PySide6)",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Conversion in progress",
                "A conversion is still running. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.request_cancel()
            self._worker.wait(3000)
        self._persist_options()
        event.accept()
