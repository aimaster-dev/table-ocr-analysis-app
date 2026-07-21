"""OCR engine wrapper (Tesseract) optimized for ruled tables."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_LANG_ALIASES = {
    "en": "eng",
    "english": "eng",
    "ch_sim": "chi_sim",
    "ch_tra": "chi_tra",
    "zh": "chi_sim",
    "chinese": "chi_sim",
    "ko": "kor",
    "jp": "jpn",
    "ja": "jpn",
}

_BORDER_NOISE = re.compile(r"^[\s\|\[\]\{\}_—\-–=~·•'\"`]+$")
_SERIAL = re.compile(r"^\d{1,3}$")


class OcrEngine:
    """Thread-safe Tesseract facade via pytesseract."""

    _lock = threading.Lock()
    _configured = False

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        tesseract_cmd: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        del use_gpu
        self.languages = [self._normalize_lang(lang) for lang in (languages or ["eng"])]
        self.tesseract_cmd = tesseract_cmd or DEFAULT_TESSERACT_CMD
        self._ready = False

    @staticmethod
    def _normalize_lang(lang: str) -> str:
        key = lang.strip().lower()
        return _LANG_ALIASES.get(key, lang.strip())

    @property
    def lang_string(self) -> str:
        return "+".join(self.languages)

    def warm_up(self) -> None:
        self._ensure_ready()

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import pytesseract

            cmd = Path(self.tesseract_cmd)
            if not cmd.is_file():
                raise FileNotFoundError(
                    f"Tesseract executable not found:\n{cmd}\n\n"
                    "Install Tesseract OCR or update the path in settings."
                )

            pytesseract.pytesseract.tesseract_cmd = str(cmd)
            version = pytesseract.get_tesseract_version()
            logger.info(
                "Tesseract ready version=%s lang=%s cmd=%s",
                version,
                self.lang_string,
                cmd,
            )
            installed = set(pytesseract.get_languages(config=""))
            missing = [lang for lang in self.languages if lang not in installed]
            if missing:
                raise RuntimeError(
                    "Tesseract language data is missing: "
                    + ", ".join(missing)
                    + "\n\nInstall the requested .traineddata files in tessdata, "
                    "or use PaddleOCR for East Asian documents."
                )
            self._ready = True
            OcrEngine._configured = True

    def read_words(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        """
        Full-page OCR word boxes.

        Returns list of ``(x, y, w, h, text, confidence)`` in image coordinates.
        """
        self._ensure_ready()
        import pytesseract
        from pytesseract import Output

        # Upscale the whole page once — helps tiny serial digits a lot.
        scale = 2.0
        working = self._upscale(image_bgr, scale)
        rgb = self._to_rgb(working)

        data = pytesseract.image_to_data(
            rgb,
            lang=self.lang_string,
            output_type=Output.DICT,
            config="--oem 3 --psm 6",
        )

        words: list[tuple[int, int, int, int, str, float]] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = self.clean_text(str(data["text"][i] or ""))
            if not text:
                continue
            try:
                conf_raw = float(data["conf"][i])
            except (TypeError, ValueError):
                conf_raw = -1.0
            if conf_raw < 0:
                continue

            x = int(round(int(data["left"][i]) / scale))
            y = int(round(int(data["top"][i]) / scale))
            w = int(round(int(data["width"][i]) / scale))
            h = int(round(int(data["height"][i]) / scale))
            words.append((x, y, w, h, text, conf_raw / 100.0))
        return words

    def read_text(self, image: np.ndarray, *, detail: int = 1) -> list[Any]:
        """EasyOCR-compatible tuples for fallback layout clustering."""
        del detail
        results: list[Any] = []
        for x, y, w, h, text, conf in self.read_words(image):
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append((box, text, conf))
        return results

    def read_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        """OCR a single cell crop."""
        if cell_bgr is None or cell_bgr.size == 0:
            return "", 0.0

        h, w = cell_bgr.shape[:2]
        if h < 6 or w < 6:
            return "", 0.0

        self._ensure_ready()
        import pytesseract
        from pytesseract import Output

        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY) if cell_bgr.ndim == 3 else cell_bgr
        scale = max(3.0, 56.0 / max(h, 1))
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        up = cv2.copyMakeBorder(up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

        # Tall cells commonly contain wrapped or stacked text even when the
        # column itself is narrow.  Single-line mode was dropping the first
        # line of labels such as ``1. Time\nManagement``.
        if h >= 45:
            configs = ["--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 7"]
        elif w / max(h, 1) < 1.8:
            # Never apply a numeric whitelist to an unknown cell: a narrow
            # column can legitimately contain one CJK character or a unit.
            configs = ["--oem 3 --psm 7", "--oem 3 --psm 8", "--oem 3 --psm 10"]
        else:
            configs = ["--oem 3 --psm 7", "--oem 3 --psm 6"]

        best_text = ""
        best_conf = 0.0
        best_score = -1.0
        for config in configs:
            data = pytesseract.image_to_data(
                up,
                lang=self.lang_string,
                output_type=Output.DICT,
                config=config,
            )
            line_parts: dict[tuple[int, int, int], list[str]] = {}
            weighted_confidence = 0.0
            weight = 0
            for index, raw_text in enumerate(data.get("text", [])):
                token = self.clean_text(str(raw_text or ""))
                if not token:
                    continue
                try:
                    confidence = float(data["conf"][index]) / 100.0
                except (KeyError, TypeError, ValueError):
                    confidence = 0.0
                if confidence < 0.0:
                    continue
                key = (
                    int(data.get("block_num", [0])[index]),
                    int(data.get("par_num", [0])[index]),
                    int(data.get("line_num", [0])[index]),
                )
                line_parts.setdefault(key, []).append(token)
                token_weight = max(len(token), 1)
                weighted_confidence += confidence * token_weight
                weight += token_weight
            text = "\n".join(" ".join(parts) for parts in line_parts.values())
            text = self.clean_text(text)
            if not text:
                continue
            confidence = weighted_confidence / weight if weight else 0.0
            # Confidence is primary.  Length is only a small tie-breaker so a
            # long hallucination cannot beat a short, high-confidence label.
            score = confidence + min(len(text), 40) * 0.001
            if score > best_score:
                best_text = text
                best_conf = confidence
                best_score = score
                if confidence >= 0.92 and len(text) >= 3:
                    break

        if not best_text:
            return "", 0.0
        return best_text, best_conf

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        text = " ".join(text.replace("\n", " ").split()).strip()
        if not text or _BORDER_NOISE.match(text):
            return ""
        return text

    @staticmethod
    def normalize_cell_value(text: str, *, col_index: int, col_count: int) -> str:
        """Apply only lossless, domain-neutral cell cleanup.

        Earlier versions guessed that the first column was a serial number and
        every middle numeric-looking cell was a price.  Those guesses silently
        changed IDs, dates, measurements, and East Asian text.  Table Scan has
        no knowledge of a document's business schema, so it must preserve OCR
        output instead of inventing or rewriting values.
        """
        del col_index, col_count
        text = OcrEngine._normalize_bullet_layout(str(text))
        # Preserve page-level line grouping so bullet lists and wrapped notes
        # remain readable in Excel.  Clean each line independently because the
        # general OCR token cleaner intentionally collapses whitespace.
        lines = [OcrEngine.clean_text(line) for line in str(text).splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _normalize_bullet_layout(text: str) -> str:
        """Restore a list only when OCR found several bullet-like markers."""
        text = unicodedata.normalize("NFKC", text)
        marker = re.compile(r"(^|\s)([+*•«·])\s*(?=[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])")
        matches = list(marker.finditer(text))
        if len(matches) < 3:
            return text

        def replace(match: re.Match[str]) -> str:
            prefix = "" if match.start() == 0 else "\n"
            return prefix + "• "

        return marker.sub(replace, text)

    @staticmethod
    def _upscale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-3:
            return image_bgr
        return cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 2:
            return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
