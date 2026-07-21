"""Application settings and defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

from table_scan.utils.resource_path import user_data_dir

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

DEFAULT_TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
DEFAULT_PADDLE_LANG = "mixed_ko_zh_en"

OCR_ENGINE_LOCAL = "local"
OCR_ENGINE_PADDLE = "paddle"
OCR_ENGINE_VLM = "vlm"
OCR_ENGINE_URL = "url"

OUTPUT_FORMAT_EXCEL = "excel"
OUTPUT_FORMAT_HTML = "html"
OUTPUT_FORMAT_BOTH = "both"
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_BOTH
SUPPORTED_OUTPUT_FORMATS = frozenset(
    {OUTPUT_FORMAT_EXCEL, OUTPUT_FORMAT_HTML, OUTPUT_FORMAT_BOTH}
)


@dataclass
class AppSettings:
    """Persisted user preferences."""

    settings_version: int = 3
    last_input_dir: str = ""
    last_output_dir: str = ""
    output_format: str = DEFAULT_OUTPUT_FORMAT
    ocr_languages: list[str] = field(default_factory=lambda: ["eng"])
    # PaddleOCR is the safest first-run choice for the app's primary CJK use
    # case and also recognizes Latin text and numbers in the selected model.
    ocr_engine: str = OCR_ENGINE_PADDLE
    # Local: Tesseract install dir / exe.
    # VLM: Ollama base URL (default localhost).
    # URL: remote OCR HTTP endpoint.
    # Paddle: unused (models auto-cached); kept for UI consistency.
    ocr_location: str = DEFAULT_TESSERACT_DIR
    vlm_model: str = DEFAULT_VLM_MODEL
    paddle_lang: str = DEFAULT_PADDLE_LANG
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD
    use_gpu: bool = False
    deskew: bool = True
    rectify_perspective: bool = True
    enhance_contrast: bool = True
    min_cell_confidence: float = 0.3

    @classmethod
    def _config_path(cls) -> Path:
        return user_data_dir() / "settings.json"

    @classmethod
    def load(cls) -> AppSettings:
        path = cls._config_path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_version = int(data.get("settings_version", 1))
            known = set(cls.__dataclass_fields__)
            filtered = {k: v for k, v in data.items() if k in known}
            langs = filtered.get("ocr_languages")
            if isinstance(langs, list):
                filtered["ocr_languages"] = [
                    "eng" if str(x).lower() in {"en", "english"} else str(x) for x in langs
                ]
            settings = cls(**filtered)
            if stored_version < 2:
                if (
                    settings.ocr_engine == OCR_ENGINE_PADDLE
                    and settings.paddle_lang == "ch"
                ):
                    settings.paddle_lang = DEFAULT_PADDLE_LANG
            # Existing installations keep their previous Excel-only behavior;
            # new installations default to Both so HTML can be reviewed next
            # to the workbook without running OCR a second time.
            if stored_version < 3 and "output_format" not in data:
                settings.output_format = OUTPUT_FORMAT_EXCEL
            settings.output_format = cls.normalize_output_format(settings.output_format)
            settings.settings_version = 3
            settings._migrate_location()
            return settings
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def _migrate_location(self) -> None:
        if self.ocr_engine == OCR_ENGINE_VLM:
            if not self.ocr_location or not self.ocr_location.lower().startswith("http"):
                self.ocr_location = DEFAULT_OLLAMA_URL
            return
        if self.ocr_engine == OCR_ENGINE_PADDLE:
            return
        if self.ocr_location and self.ocr_location != DEFAULT_TESSERACT_DIR:
            return
        if self.tesseract_cmd and self.tesseract_cmd != DEFAULT_TESSERACT_CMD:
            path = Path(self.tesseract_cmd)
            self.ocr_location = str(path.parent if path.suffix.lower() == ".exe" else path)
            self.ocr_engine = OCR_ENGINE_LOCAL

    @staticmethod
    def normalize_output_format(value: str | None) -> str:
        normalized = str(value or "").strip().lower()
        return (
            normalized
            if normalized in SUPPORTED_OUTPUT_FORMATS
            else DEFAULT_OUTPUT_FORMAT
        )

    def save(self) -> None:
        self.output_format = self.normalize_output_format(self.output_format)
        self.settings_version = 3
        if self.ocr_engine == OCR_ENGINE_LOCAL:
            from table_scan.core.tesseract_paths import resolve_tesseract_cmd

            try:
                self.tesseract_cmd = str(resolve_tesseract_cmd(self.ocr_location))
            except FileNotFoundError:
                pass
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
