"""Offline vision-language OCR via local Ollama (no cloud)."""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from table_scan.models.table_result import ExtractedTable

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"

_TABLE_PROMPT = """You are an offline document OCR engine. Read the table in this image carefully,
including handwritten text. Return ONLY valid JSON with this shape:

{"rows":[["cell","cell",...],["cell","cell",...],...]}

Rules:
- Preserve every row and column of the main table.
- Use empty string "" for blank cells.
- Do not invent values that are not visible.
- Do not wrap the JSON in markdown fences.
- Do not add commentary before or after the JSON.
"""


class VlmOcrEngine:
    """
    Local vision-language model through Ollama's HTTP API.

    Images are sent only to the configured host (default ``127.0.0.1``),
    so nothing is uploaded to the public internet when Ollama runs locally.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_OLLAMA_URL,
        *,
        model: str = DEFAULT_VLM_MODEL,
        languages: list[str] | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        del languages  # prompt is language-agnostic; kept for factory symmetry
        self.endpoint = endpoint.rstrip("/")
        self.model = model.strip() or DEFAULT_VLM_MODEL
        self.timeout_seconds = timeout_seconds

    def warm_up(self) -> None:
        tags = self._get_json(f"{self.endpoint}/api/tags")
        models = [m.get("name", "") for m in tags.get("models", []) if isinstance(m, dict)]
        logger.info("Ollama online at %s (%s model(s))", self.endpoint, len(models))
        if not any(self._model_matches(name) for name in models):
            raise RuntimeError(
                f"Ollama is running, but model '{self.model}' was not found.\n\n"
                f"Install it offline with:\n"
                f"  ollama pull {self.model}\n\n"
                f"Available: {', '.join(models) or '(none)'}"
            )

    def extract_table(self, image_bgr: np.ndarray) -> ExtractedTable:
        """Ask the VLM to return the full table as JSON rows."""
        raw = self._chat_with_image(image_bgr, _TABLE_PROMPT)
        rows = self._parse_rows(raw)
        if not rows:
            raise ValueError(
                "Local VLM returned no table rows.\n"
                f"Raw response (truncated): {raw[:400]!r}"
            )
        return ExtractedTable(rows=rows)

    def read_words(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        """Fallback: treat whole-page transcription as one block."""
        table = self.extract_table(image_bgr)
        h, w = image_bgr.shape[:2]
        text = " | ".join(" ".join(row) for row in table.rows)
        return [(0, 0, w, h, text, 0.8)] if text.strip() else []

    def read_text(self, image: np.ndarray, *, detail: int = 1) -> list[Any]:
        del detail
        results: list[Any] = []
        for x, y, w, h, text, conf in self.read_words(image):
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append((box, text, conf))
        return results

    def read_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        prompt = (
            "Transcribe all text in this image cell, including handwriting. "
            "Return ONLY the plain text, nothing else."
        )
        text = self._chat_with_image(cell_bgr, prompt).strip()
        text = text.strip("`").strip()
        return text, 0.8 if text else 0.0

    def _model_matches(self, available: str) -> bool:
        wanted = self.model.lower()
        name = available.lower()
        return name == wanted or name.startswith(wanted + ":") or wanted.startswith(name)

    def _chat_with_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image for local VLM")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "options": {
                "temperature": 0.1,
            },
        }
        data = self._post_json(f"{self.endpoint}/api/chat", payload)
        message = data.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            # Older generate-style fallback key
            content = data.get("response")
        if not content:
            raise RuntimeError(f"Empty response from Ollama model '{self.model}'")
        return str(content)

    def _parse_rows(self, raw: str) -> list[list[str]]:
        text = raw.strip()
        # Strip ```json ... ``` if the model ignored instructions.
        fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1)
        else:
            # Take the outermost JSON object/array.
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("VLM JSON parse failed; attempting row repair")
            return self._heuristic_rows(raw)

        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("table") or payload.get("data")
        else:
            rows = payload

        if not isinstance(rows, list):
            return []

        cleaned: list[list[str]] = []
        for row in rows:
            if isinstance(row, list):
                cleaned.append([("" if c is None else str(c)).strip() for c in row])
            elif isinstance(row, dict):
                cleaned.append([str(v).strip() for v in row.values()])
            else:
                cleaned.append([str(row).strip()])
        return [r for r in cleaned if any(cell for cell in r)]

    @staticmethod
    def _heuristic_rows(raw: str) -> list[list[str]]:
        """Last-resort: split plain text into pipe/tab rows."""
        rows: list[list[str]] = []
        for line in raw.splitlines():
            line = line.strip().strip("|")
            if not line or line.startswith("```"):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.split("|")]
            elif "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                continue
            if any(cells):
                rows.append(cells)
        return rows

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Cannot reach local Ollama.\n\n"
                f"URL: {url}\n"
                f"Error: {exc}\n\n"
                "Install Ollama from https://ollama.com and keep it running offline.\n"
                f"Then: ollama pull {self.model}"
            ) from exc

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Local VLM request failed.\nURL: {url}\nModel: {self.model}\n{exc}"
            ) from exc
