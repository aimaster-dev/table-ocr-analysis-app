"""Remote HTTP OCR engine (optional alternative to local Tesseract)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from table_scan.core.ocr_engine import OcrEngine

logger = logging.getLogger(__name__)


class HttpOcrEngine:
    """
    Calls a remote OCR HTTP endpoint.

    Expected request: ``POST`` multipart field ``image`` (PNG bytes).
    Expected JSON response (preferred)::

        {
          "words": [
            {"text": "Hello", "x": 10, "y": 20, "w": 40, "h": 12, "confidence": 0.9}
          ]
        }

    Or plain text body (fallback) — treated as a single full-page string.
    """

    def __init__(self, endpoint: str, languages: list[str] | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.languages = languages or ["eng"]

    def warm_up(self) -> None:
        logger.info("HTTP OCR endpoint: %s", self.endpoint)

    def read_words(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        payload = self._post_image(image_bgr)
        words: list[tuple[int, int, int, int, str, float]] = []

        if isinstance(payload, dict) and isinstance(payload.get("words"), list):
            for item in payload["words"]:
                if not isinstance(item, dict):
                    continue
                text = OcrEngine.clean_text(str(item.get("text", "")))
                if not text:
                    continue
                conf = float(item.get("confidence", item.get("conf", 0.8)))
                if conf > 1.0:
                    conf = conf / 100.0
                words.append(
                    (
                        int(item.get("x", 0)),
                        int(item.get("y", 0)),
                        int(item.get("w", item.get("width", 0))),
                        int(item.get("h", item.get("height", 0))),
                        text,
                        conf,
                    )
                )
            return words

        if isinstance(payload, dict) and "text" in payload:
            text = OcrEngine.clean_text(str(payload.get("text", "")))
            if text:
                h, w = image_bgr.shape[:2]
                return [(0, 0, w, h, text, 0.5)]
            return []

        if isinstance(payload, str):
            text = OcrEngine.clean_text(payload)
            if text:
                h, w = image_bgr.shape[:2]
                return [(0, 0, w, h, text, 0.5)]
        return []

    def read_text(self, image: np.ndarray, *, detail: int = 1) -> list[Any]:
        del detail
        results: list[Any] = []
        for x, y, w, h, text, conf in self.read_words(image):
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append((box, text, conf))
        return results

    def read_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        words = self.read_words(cell_bgr)
        if not words:
            return "", 0.0
        text = OcrEngine.clean_text(" ".join(w[4] for w in words))
        conf = sum(w[5] for w in words) / len(words)
        return text, conf

    def _post_image(self, image_bgr: np.ndarray) -> Any:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image for HTTP OCR")

        boundary = "----TableScanBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="table.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + encoded.tobytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json, text/plain",
                "X-OCR-Languages": "+".join(self.languages),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP OCR request failed:\n{self.endpoint}\n{exc}") from exc

        text = raw.decode("utf-8", errors="replace")
        if "json" in content_type.lower() or text.lstrip().startswith(("{", "[")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
