"""Offline PaddleOCR engine — strong for East Asian print & handwriting."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# UI label → PaddleOCR lang code
MIXED_HANDWRITING_LANG = "mixed_ko_zh_en"
MIXED_HANDWRITING_MODEL = "PP-OCRv6_medium_rec"
PADDLE_LANG_CHOICES: list[tuple[str, str]] = [
    (
        "Handwritten Korean + Chinese + English / numbers (high accuracy)",
        MIXED_HANDWRITING_LANG,
    ),
    ("Chinese Simplified + English / numbers", "ch"),
    ("Chinese Traditional + English / numbers", "chinese_cht"),
    ("Japanese + English / numbers", "japan"),
    ("Korean + English / numbers", "korean"),
    ("English + numbers", "en"),
]

DEFAULT_PADDLE_LANG = MIXED_HANDWRITING_LANG


class PaddleOcrEngine:
    """
    Local PaddleOCR facade.

    Models download once to the user cache, then run fully offline.
    Best default for East Asian handwriting among classic OCR engines.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        use_gpu: bool = False,
        paddle_lang: str | None = None,
    ) -> None:
        self.paddle_lang = self._resolve_lang(paddle_lang, languages)
        self.mixed_handwriting = self.paddle_lang == MIXED_HANDWRITING_LANG
        self.use_gpu = use_gpu
        self._ocr: Any | None = None

    @staticmethod
    def _resolve_lang(paddle_lang: str | None, languages: list[str] | None) -> str:
        if paddle_lang:
            return paddle_lang.strip()
        if not languages:
            return DEFAULT_PADDLE_LANG
        raw = str(languages[0]).strip().lower()
        aliases = {
            "ch": "ch",
            "chinese": "ch",
            "zh": "ch",
            "zh-cn": "ch",
            "chi_sim": "ch",
            "chinese_cht": "chinese_cht",
            "zh-tw": "chinese_cht",
            "zh-hk": "chinese_cht",
            "chi_tra": "chinese_cht",
            "japan": "japan",
            "japanese": "japan",
            "ja": "japan",
            "jpn": "japan",
            "korean": "korean",
            "ko": "korean",
            "kor": "korean",
            "en": "en",
            "eng": "en",
            "english": "en",
        }
        return aliases.get(raw, raw or DEFAULT_PADDLE_LANG)

    def warm_up(self) -> None:
        self._ensure_ocr()

    def _ensure_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        with self._lock:
            if self._ocr is not None:
                return self._ocr
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed.\n\n"
                    "Install offline packages:\n"
                    "  pip install \"paddlepaddle==3.2.2\" \"paddleocr==3.6.0\"\n"
                ) from exc

            logger.info("Loading PaddleOCR lang=%s gpu=%s", self.paddle_lang, self.use_gpu)
            cpu_threads = max(1, min(8, os.cpu_count() or 4))
            # PaddlePaddle 3.3.x has a known PIR -> oneDNN attribute conversion
            # failure on CPU.  OCR correctness matters more than this optional
            # acceleration, so keep it disabled on CPU until upstream fixes it.
            enable_mkldnn = False
            if self.mixed_handwriting:
                # PP-OCRv6 uses one shared character vocabulary for Korean,
                # Chinese, Latin text, and digits.  Separate language models
                # cannot safely recognize a cell containing multiple scripts.
                init_attempts: list[dict[str, Any]] = [
                    {
                        "text_recognition_model_name": MIXED_HANDWRITING_MODEL,
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": True,
                        "text_recognition_batch_size": 6,
                        "device": "gpu" if self.use_gpu else "cpu",
                        "enable_mkldnn": enable_mkldnn,
                        "cpu_threads": cpu_threads,
                    }
                ]
            else:
                # Support both older and newer PaddleOCR constructors for the
                # language-specific lightweight modes.
                init_attempts = [
                    {
                        "lang": self.paddle_lang,
                        "use_textline_orientation": True,
                        "device": "gpu" if self.use_gpu else "cpu",
                        "enable_mkldnn": enable_mkldnn,
                        "cpu_threads": cpu_threads,
                    },
                    {
                        "lang": self.paddle_lang,
                        "use_angle_cls": True,
                        "use_gpu": self.use_gpu,
                        "show_log": False,
                        "enable_mkldnn": enable_mkldnn,
                        "cpu_threads": cpu_threads,
                    },
                    {
                        "lang": self.paddle_lang,
                        "use_angle_cls": True,
                    },
                ]
            last_error: Exception | None = None
            for kwargs in init_attempts:
                try:
                    self._ocr = PaddleOCR(**kwargs)
                    logger.info("PaddleOCR ready with %s", kwargs)
                    return self._ocr
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    # Device/gpu kwargs may fail on CPU-only builds; try next.
                    continue
            if self.mixed_handwriting:
                raise RuntimeError(
                    "The mixed Korean/Chinese handwriting mode requires "
                    "PaddleOCR 3.6 or newer and PaddlePaddle 3.2 or newer.\n\n"
                    "Install the tested versions with:\n"
                    "  pip install --upgrade --force-reinstall "
                    "\"paddlepaddle==3.2.2\" \"paddleocr==3.6.0\"\n\n"
                    f"Initialization error: {last_error}"
                )
            raise RuntimeError(f"Failed to initialize PaddleOCR: {last_error}")

    def read_words(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int, str, float]]:
        lines = self._run_ocr(image_bgr)
        words: list[tuple[int, int, int, int, str, float]] = []
        for box, text, conf in lines:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            words.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), text, conf))
        return words

    def read_text(self, image: np.ndarray, *, detail: int = 1) -> list[Any]:
        del detail
        working, scale = self._prepare_page(image)
        results: list[Any] = []
        for box, text, conf in self._run_ocr(working):
            if scale > 1.0:
                box = [[point[0] / scale, point[1] / scale] for point in box]
            results.append((box, text, conf))
        return results

    def read_cell(self, cell_bgr: np.ndarray) -> tuple[str, float]:
        if cell_bgr is None or cell_bgr.size == 0:
            return "", 0.0
        h, w = cell_bgr.shape[:2]
        if h < 6 or w < 6:
            return "", 0.0

        # Upscale tiny cells — helps handwritten CJK strokes.
        target_height = 96.0 if self.mixed_handwriting else 64.0
        scale = min(6.0, max(2.0, target_height / max(h, 1)))
        if scale > 1.05:
            cell_bgr = cv2.resize(
                cell_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )

        lines = self._run_ocr(cell_bgr)
        text, conf = self._join_lines(lines)

        # Faint pencil and uneven pen pressure often benefit from a clean
        # high-contrast retry.  It runs only for weak mixed-handwriting cells.
        if self.mixed_handwriting and conf < 0.72:
            binary = self._handwriting_binary(cell_bgr)
            retry_text, retry_conf = self._join_lines(self._run_ocr(binary))
            if retry_text and (
                not text
                or retry_conf > conf + 0.02
                or (retry_conf >= conf and len(retry_text) > len(text))
            ):
                text, conf = retry_text, retry_conf
        return text, conf

    def _prepare_page(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.mixed_handwriting or image.size == 0:
            return image, 1.0
        height, width = image.shape[:2]
        scale = min(3.0, max(1.0, 1800.0 / max(height, width, 1)))
        if scale <= 1.05:
            return image, 1.0
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
        return resized, scale

    @staticmethod
    def _join_lines(
        lines: list[tuple[list[list[float]], str, float]],
    ) -> tuple[str, float]:
        if not lines:
            return "", 0.0
        ordered = sorted(
            lines,
            key=lambda item: (
                min(point[1] for point in item[0]),
                min(point[0] for point in item[0]),
            ),
        )
        text = " ".join(value for _, value, _ in ordered).strip()
        weight = sum(max(len(value), 1) for _, value, _ in ordered)
        confidence = sum(
            score * max(len(value), 1) for _, value, score in ordered
        ) / weight
        return text, confidence

    @staticmethod
    def _handwriting_binary(image_bgr: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_bgr.ndim == 3
            else image_bgr
        )
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _run_ocr(self, image_bgr: np.ndarray) -> list[tuple[list[list[float]], str, float]]:
        ocr = self._ensure_ocr()
        # PaddleOCR's ndarray API follows OpenCV's BGR convention.  Swapping to
        # RGB reduces contrast on colored forms and is unnecessary for scans.
        working = image_bgr

        # PaddleOCR 2.x: .ocr(); 3.x: .predict() / compatibility .ocr().
        errors: list[str] = []
        for method_name, kwargs in (
            ("predict", {}),
            ("ocr", {"cls": True}),
            ("ocr", {}),
        ):
            method = getattr(ocr, method_name, None)
            if method is None:
                continue
            try:
                raw = method(working, **kwargs) if kwargs else method(working)
                return self._normalize_result(raw)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
                continue

        detail = "; ".join(errors[-3:]) or "no compatible OCR method"
        if (
            "ConvertPirAttribute2RuntimeAttribute" in detail
            or "onednn_instruction" in detail
        ):
            raise RuntimeError(
                "PaddleOCR hit a known Windows CPU/oneDNN compatibility error.\n\n"
                "Inside the activated virtual environment, run:\n"
                "  python -m pip install --upgrade --force-reinstall "
                "\"paddlepaddle==3.2.2\" \"paddleocr==3.6.0\"\n\n"
                "Then restart Table Scan."
            )
        raise RuntimeError(f"PaddleOCR inference failed ({detail})")

    @staticmethod
    def _normalize_result(
        raw: Any,
        _depth: int = 0,
    ) -> list[tuple[list[list[float]], str, float]]:
        """Normalize various PaddleOCR result shapes into (box, text, conf)."""
        if raw is None or _depth > 8:
            return []

        if isinstance(raw, str):
            try:
                return PaddleOcrEngine._normalize_result(json.loads(raw), _depth + 1)
            except json.JSONDecodeError:
                return []

        if isinstance(raw, dict):
            return PaddleOcrEngine._normalize_dict_result(raw, _depth=_depth + 1)

        parsed = PaddleOcrEngine._parse_line(raw)
        if parsed is not None:
            return [parsed]

        if isinstance(raw, (list, tuple)):
            items: list[tuple[list[list[float]], str, float]] = []
            for entry in raw:
                items.extend(PaddleOcrEngine._normalize_result(entry, _depth + 1))
            return items

        # Object with attributes (OCRResult)
        for attr in ("json", "res", "ocr_result"):
            if hasattr(raw, attr):
                value = getattr(raw, attr)
                if callable(value):
                    try:
                        value = value()
                    except Exception:  # noqa: BLE001
                        continue
                normalized = PaddleOcrEngine._normalize_result(value, _depth + 1)
                if normalized:
                    return normalized

        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            items: list[tuple[list[list[float]], str, float]] = []
            for entry in raw:
                items.extend(PaddleOcrEngine._normalize_result(entry, _depth + 1))
            return items

        return []

    @staticmethod
    def _normalize_dict_result(
        data: dict[str, Any],
        *,
        _depth: int = 0,
    ) -> list[tuple[list[list[float]], str, float]]:
        items: list[tuple[list[list[float]], str, float]] = []
        # Common 3.x keys
        texts = PaddleOcrEngine._first_present(data, "rec_texts", "texts")
        scores = PaddleOcrEngine._first_present(data, "rec_scores", "scores")
        polys = PaddleOcrEngine._first_present(
            data, "dt_polys", "rec_polys", "rec_boxes", "boxes"
        )
        texts = [] if texts is None else list(texts)
        scores = [] if scores is None else list(scores)
        polys = [] if polys is None else list(polys)
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.8
            poly = polys[i] if i < len(polys) else [[0, 0], [1, 0], [1, 1], [0, 1]]
            box = PaddleOcrEngine._box_points(poly)
            if box is None:
                continue
            text_s = str(text).strip()
            if text_s:
                items.append((box, text_s, conf if conf <= 1.0 else conf / 100.0))
        if items:
            return items

        # Some v3 objects wrap the recognition dictionary in ``res``.
        for key in ("res", "ocr_result", "data", "result"):
            if key in data:
                normalized = PaddleOcrEngine._normalize_result(data[key], _depth + 1)
                if normalized:
                    return normalized
        return []

    @staticmethod
    def _parse_line(line: Any) -> tuple[list[list[float]], str, float] | None:
        # [box, (text, conf)]
        if isinstance(line, (list, tuple)) and len(line) >= 2:
            box = line[0]
            info = line[1]
            if isinstance(info, (list, tuple)) and len(info) >= 2:
                if not isinstance(info[0], str):
                    return None
                text = info[0].strip()
                try:
                    conf = float(info[1])
                except (TypeError, ValueError):
                    return None
                if not text:
                    return None
                points = PaddleOcrEngine._box_points(box)
                if points is None:
                    return None
                return points, text, conf if conf <= 1.0 else conf / 100.0
        return None

    @staticmethod
    def _first_present(data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    @staticmethod
    def _box_points(box: Any) -> list[list[float]] | None:
        try:
            array = np.asarray(box, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.ndim == 1 and array.size == 4:
            x1, y1, x2, y2 = array.tolist()
            array = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        try:
            array = array.reshape(-1, 2)
        except ValueError:
            return None
        if len(array) < 4 or not np.isfinite(array).all():
            return None
        return [[float(point[0]), float(point[1])] for point in array]
