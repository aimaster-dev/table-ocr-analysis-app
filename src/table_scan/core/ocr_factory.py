"""Factory for Tesseract / PaddleOCR / offline VLM / remote HTTP OCR engines."""

from __future__ import annotations

from table_scan.config.settings import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_PADDLE_LANG,
    DEFAULT_VLM_MODEL,
    OCR_ENGINE_LOCAL,
    OCR_ENGINE_PADDLE,
    OCR_ENGINE_URL,
    OCR_ENGINE_VLM,
    AppSettings,
)
from table_scan.core.http_ocr_engine import HttpOcrEngine
from table_scan.core.ocr_engine import OcrEngine
from table_scan.core.paddle_ocr_engine import PaddleOcrEngine
from table_scan.core.tesseract_paths import apply_tessdata_env, resolve_tesseract_cmd
from table_scan.core.vlm_ocr_engine import VlmOcrEngine


def create_ocr_engine(settings: AppSettings):
    engine = (settings.ocr_engine or OCR_ENGINE_LOCAL).strip().lower()
    location = (settings.ocr_location or "").strip()

    if engine == OCR_ENGINE_PADDLE:
        return PaddleOcrEngine(
            settings.ocr_languages,
            use_gpu=settings.use_gpu,
            paddle_lang=settings.paddle_lang or DEFAULT_PADDLE_LANG,
            model_dir=settings.paddle_model_dir,
        )

    if engine == OCR_ENGINE_VLM:
        endpoint = location or DEFAULT_OLLAMA_URL
        if not endpoint.lower().startswith(("http://", "https://")):
            raise ValueError(
                "Local VLM expects an Ollama URL, e.g. http://127.0.0.1:11434\n"
                f"Got: {endpoint}"
            )
        return VlmOcrEngine(
            endpoint,
            model=settings.vlm_model or DEFAULT_VLM_MODEL,
            languages=settings.ocr_languages,
        )

    if engine == OCR_ENGINE_URL:
        if not location:
            raise ValueError("OCR URL is empty. Enter a remote OCR endpoint.")
        if not location.lower().startswith(("http://", "https://")):
            raise ValueError(
                "OCR engine is set to URL, but the location is not an http(s) address.\n"
                f"Got: {location}"
            )
        return HttpOcrEngine(location, languages=settings.ocr_languages)

    if not location:
        raise ValueError("OCR engine directory is empty. Browse to your Tesseract-OCR folder.")
    cmd = resolve_tesseract_cmd(location)
    apply_tessdata_env(location)
    return OcrEngine(
        settings.ocr_languages,
        tesseract_cmd=str(cmd),
        use_gpu=settings.use_gpu,
    )
